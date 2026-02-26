from cls_rmit_spot_core import SpotAgent

if __name__ == "__main__":
    with SpotAgent(
        stream=True,
        navigation=True
        ) as agent:
        agent.power_on_and_stand()
        agent.shutdown(power_off=False)
