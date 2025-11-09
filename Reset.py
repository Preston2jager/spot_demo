from cls_spot_agent import SpotAgent

if __name__ == "__main__":
    agent = SpotAgent(
        hostname="192.168.80.3",
        username="user",
        password="myjujz7e2prj",
        force_lease=True,
        client_name="MySpotClient",
    )

    #agent.setup_software_estop()   # 新增：建立软件 E-Stop keep-alive
    agent.power_on_and_stand()

    agent.shutdown(power_off=False)
