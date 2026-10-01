"""Play audio through the USB speaker plugged into Spot's CoreIO.

CoreIO ships only ALSA's `aplay`/`paplay` — there is no mp3 decoder on the box — so
source audio is transcoded to plain PCM WAV locally with ffmpeg, pushed over SFTP, and
played remotely over SSH.
"""
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Self

import paramiko
from loguru import logger

_PROJECT_ROOT = Path(__file__).parent.parent
# Audio lives with the other shared, checked-in inputs rather than beside either front
# end: the display and control pages play the same files through one agent.
AUDIO_DIR = _PROJECT_ROOT / "assets" / "audio"
_REMOTE_DIR = "/tmp/spot_speaker"

# The C-Media USB dongle advertises S16_LE / 2ch / 44100 or 48000. Normalising to the
# top of that range means `plughw` never has to resample on the robot's weaker CPU.
_WAV_RATE = 48000
_WAV_CHANNELS = 2

# ALSA prints this whenever it cannot open the PCM at all — an unplugged USB dongle, a
# renamed card, a bad DEV index. Any other aplay failure is a problem with the audio.
_DEVICE_MISSING_MARKER = "audio open error"

# USB dongles name their playback control differently; the first of these that the card
# actually exposes is the one that moves the speaker.
_MIXER_CONTROL_PREFERENCE = ("PCM", "Speaker", "Master", "Headphone")


class SpeakerDeviceMissing(RuntimeError):
    """The CoreIO has no ALSA playback device matching the configured name."""


def _parse_volume(value: int | str | None) -> int | None:
    """Read a volume percentage, clamped to 0–100. Blank or unusable input means "leave it"."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        return max(0, min(100, int(str(value).strip().rstrip("%"))))
    except ValueError:
        logger.warning(f"CoreIO speaker: ignoring COREIO_SPEAKER_VOLUME='{value}' — expected 0-100")
        return None


def resolve_audio(audio: str | Path) -> Path:
    """Resolve an audio path and verify it exists.

    Two roots, tried in that order: the project root, so a configured path like
    ``assets/audio/init.mp3`` means what it says, and ``assets/audio`` itself, so a bare
    ``init.mp3`` finds the shared folder. The second is what lets the web console and the
    terminal menu name the same announcement without either of them owning the audio.
    """
    source = Path(audio)
    if source.is_absolute():
        candidates = [source]
    else:
        candidates = [_PROJECT_ROOT / source, AUDIO_DIR / source]

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = " or ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Audio file not found: {tried}")


class CoreIOSpeaker:
    """SSH connection that plays local audio files on the CoreIO's USB speaker.

    Mission code should not touch this directly — use ``SpotAgent.play()``, which owns a
    lazily-opened session and swallows failures. Standalone use is as a context manager::

        with CoreIOSpeaker() as speaker:
            speaker.play("assets/audio/announce.mp3")

    Connection details default to the ``COREIO_*`` variables in ``.env``.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        username: str | None = None,
        password: str | None = None,
        device: str | None = None,
        volume: int | str | None = None,
    ) -> None:
        self.host = host or os.getenv("COREIO_HOST", "192.168.80.3")
        self.port = int(port or os.getenv("COREIO_PORT", "20022"))
        self.username = username or os.getenv("COREIO_USER", "spot")
        self.password = password or os.getenv("COREIO_PASSWORD", "")
        # `CARD=Device` addresses the USB dongle by name, which survives the card index
        # shifting when other USB audio gear is plugged or unplugged.
        self.device = device or os.getenv("COREIO_AUDIO_DEVICE", "plughw:CARD=Device,DEV=0")
        # Playback volume in percent of the card's own range. ``None`` — the default when
        # COREIO_SPEAKER_VOLUME is unset or blank — leaves the mixer exactly as it is.
        self.volume = _parse_volume(volume if volume is not None else os.getenv("COREIO_SPEAKER_VOLUME"))
        self._client: paramiko.SSHClient | None = None
        self._mixer_control: str | None = None
        self._volume_unavailable = False

    # ── connection ────────────────────────────────────────────────────────────

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def connect(self, timeout: float = 15.0) -> None:
        """Open the SSH session to the CoreIO."""
        if not self.password:
            raise RuntimeError("COREIO_PASSWORD is not set — add it to .env")
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
            timeout=timeout,
            allow_agent=False,
            look_for_keys=False,
        )
        self._client = client
        logger.debug(f"CoreIO speaker: connected to {self.username}@{self.host}:{self.port}")

    def close(self) -> None:
        """Close the SSH session if one is open."""
        if self._client is not None:
            self._client.close()
            self._client = None

    # ── playback ──────────────────────────────────────────────────────────────

    def play(self, audio: str | Path, timeout: float = 60.0) -> None:
        """Transcode, upload, and play ``audio`` on the CoreIO speaker, blocking until done.

        Relative paths resolve against the project root. ``timeout`` caps the remote
        `aplay` so a wedged playback can never stall the calling mission.
        """
        if self._client is None:
            raise RuntimeError("Not connected — call connect() or use as a context manager.")

        source = resolve_audio(audio)
        remote = f"{_REMOTE_DIR}/{self._fingerprint(source)}.wav"
        self._upload_if_missing(source, remote)
        self.apply_volume()

        cmd = f"timeout {int(timeout)} aplay -q -D {shlex.quote(self.device)} {shlex.quote(remote)}"
        logger.info(f"CoreIO speaker: playing '{source.name}' on {self.device}")
        _, stdout, stderr = self._client.exec_command(cmd)
        status = stdout.channel.recv_exit_status()
        if status != 0:
            detail = stderr.read().decode(errors="replace").strip() or f"exit status {status}"
            if _DEVICE_MISSING_MARKER in detail.lower():
                raise self._missing_device_error()
            raise RuntimeError(f"aplay failed on CoreIO: {detail}")
        logger.success(f"CoreIO speaker: finished '{source.name}'")

    def check_device(self) -> None:
        """Raise :class:`SpeakerDeviceMissing` if the configured ALSA device is not there.

        Feeding aplay an empty file opens the PCM and then immediately hits EOF, so a
        device that exists fails on the *header* while a missing one fails on the *open* —
        enough to tell "no USB sound card" from "bad audio file" without making a sound.
        """
        if self._client is None:
            raise RuntimeError("Not connected — call connect() or use as a context manager.")

        _, stdout, stderr = self._client.exec_command(
            f"aplay -q -D {shlex.quote(self.device)} /dev/null"
        )
        stdout.channel.recv_exit_status()
        if _DEVICE_MISSING_MARKER in stderr.read().decode(errors="replace").lower():
            raise self._missing_device_error()
        # Set the level now that the card is known good, so a mixer the dongle does not
        # expose is one startup warning rather than a surprise on the first announcement.
        self.apply_volume()

    # ── volume ────────────────────────────────────────────────────────────────

    def apply_volume(self) -> None:
        """Set the card's playback level to ``self.volume`` percent, if one is configured.

        Best effort and never fatal: playing at the wrong volume beats not playing at all,
        so a card without a usable mixer control is warned about once and then left alone.
        """
        if self.volume is None or self._volume_unavailable or self._client is None:
            return

        control = self._mixer_control or self._find_mixer_control()
        if control is None:
            self._volume_unavailable = True
            logger.warning(
                f"CoreIO speaker: no volume control on '{self._mixer_device()}' — "
                f"playing at the card's current level"
            )
            return

        cmd = (
            f"amixer -q -D {shlex.quote(self._mixer_device())} "
            f"sset {shlex.quote(control)} {self.volume}% unmute"
        )
        _, stdout, stderr = self._client.exec_command(cmd)
        if stdout.channel.recv_exit_status() != 0:
            self._volume_unavailable = True
            detail = stderr.read().decode(errors="replace").strip() or "amixer failed"
            logger.warning(f"CoreIO speaker: could not set volume ({detail})")
            return
        logger.debug(f"CoreIO speaker: {control} volume set to {self.volume}%")

    def _mixer_device(self) -> str:
        """The control-device name for ``self.device`` — `plughw:CARD=Device,DEV=0` → `hw:CARD=Device`.

        amixer addresses the card, not the PCM, so the `plughw`/`DEV=` parts have to go.
        A device with no card in it at all (a bare `default`) has no card to address either,
        so it is handed to amixer untouched rather than turned into a bogus `hw:default`.
        """
        if ":" not in self.device:
            return self.device
        card = self.device.split(":", 1)[1].split(",")[0]
        return f"hw:{card}"

    def _find_mixer_control(self) -> str | None:
        """Pick the card's playback control, preferring the usual USB-dongle names."""
        _, stdout, _ = self._client.exec_command(
            f"amixer -D {shlex.quote(self._mixer_device())} scontrols"
        )
        if stdout.channel.recv_exit_status() != 0:
            return None
        names = re.findall(r"Simple mixer control '([^']+)'", stdout.read().decode(errors="replace"))
        if not names:
            return None
        self._mixer_control = next((c for c in _MIXER_CONTROL_PREFERENCE if c in names), names[0])
        return self._mixer_control

    def _missing_device_error(self) -> SpeakerDeviceMissing:
        """Build the "sound card is not there" error, naming what the CoreIO does have."""
        cards = ", ".join(self.playback_cards()) or "none"
        return SpeakerDeviceMissing(
            f"no playback device '{self.device}' on the CoreIO (cards present: {cards}) — "
            f"check the USB speaker is plugged in, or set COREIO_AUDIO_DEVICE in .env"
        )

    def playback_cards(self) -> list[str]:
        """ALSA card names `aplay -l` reports on the CoreIO, in order, deduplicated."""
        _, stdout, _ = self._client.exec_command("aplay -l")
        if stdout.channel.recv_exit_status() != 0:
            return []
        names = re.findall(r"^card \d+: (\S+)", stdout.read().decode(errors="replace"), re.MULTILINE)
        return list(dict.fromkeys(names))

    # ── internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _fingerprint(source: Path) -> str:
        """Content hash of the source file plus the WAV parameters it converts to.

        Naming the remote file after this means an already-present file is by definition
        the correct one, so repeat runs skip both the transcode and the upload.
        """
        digest = hashlib.sha256(f"{_WAV_RATE}:{_WAV_CHANNELS}:".encode())
        digest.update(source.read_bytes())
        return digest.hexdigest()[:16]

    @staticmethod
    def _to_wav(source: Path, dest: Path) -> None:
        """Transcode any ffmpeg-readable audio file to 48 kHz stereo 16-bit PCM WAV."""
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on PATH — required to convert audio for CoreIO")
        proc = subprocess.run(
            [
                "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                "-i", str(source),
                "-ac", str(_WAV_CHANNELS),
                "-ar", str(_WAV_RATE),
                "-c:a", "pcm_s16le",
                str(dest),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed to convert {source.name}: {proc.stderr.strip()}")

    def _upload_if_missing(self, source: Path, remote: str) -> None:
        """Push a transcoded copy of ``source`` to ``remote``, skipping files already there."""
        sftp = self._client.open_sftp()
        try:
            try:
                sftp.stat(remote)
                logger.debug(f"CoreIO speaker: {remote} already on the robot, skipping upload")
                return
            except OSError:
                pass

            _, stdout, _ = self._client.exec_command(f"mkdir -p {shlex.quote(_REMOTE_DIR)}")
            stdout.channel.recv_exit_status()

            with tempfile.TemporaryDirectory() as tmp:
                wav = Path(tmp) / "converted.wav"
                self._to_wav(source, wav)
                size_kb = wav.stat().st_size // 1024
                logger.info(f"CoreIO speaker: uploading '{source.name}' → {remote} ({size_kb} KB)")
                sftp.put(str(wav), remote)
        finally:
            sftp.close()
