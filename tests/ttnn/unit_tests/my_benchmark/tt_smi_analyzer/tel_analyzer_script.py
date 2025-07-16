from typing import List, Dict
# from pyluwen import PciChip
import constants
from tt_tools_common.utils_common.tools_utils import detect_chips_with_callback
from rich.progress import track
import jsons
import time
import os
import pandas as pd
import pathlib

# LOG_DIR = "/home/bach/tt-smi/tel_analyzer_logs/"
LOG_DIR = pathlib.Path(__file__).parent.resolve()

class Telemetry:

    def __init__(self, devices):

        # Simplified with just one device (GS)
        self.device = devices[0]        
        self.update_telem()


    def update_telem(self):
        """Update telemetry in a given interval"""
        self.smbus_telem_info = self.get_smbus_board_info()
        self.device_telemetrys = self.get_wh_gs_chip_telemetry()

 
    def get_wh_gs_chip_telemetry(self) -> Dict:
            """Get telemetry data for GS and WH chip. None if ARC FW not running"""
            current = int(self.smbus_telem_info["TDC"], 16) & 0xFFFF
            if self.smbus_telem_info["VCORE"] is not None:
                voltage = int(self.smbus_telem_info["VCORE"], 16) / 1000
            else:
                voltage = 10000
            power = int(self.smbus_telem_info["TDP"], 16) & 0xFFFF
            asic_temperature = (
                int(self.smbus_telem_info["ASIC_TEMPERATURE"], 16) & 0xFFFF
            ) / 16
            aiclk = int(self.smbus_telem_info["AICLK"], 16) & 0xFFFF

            chip_telemetry = {
                "voltage": voltage,
                "current": current,
                "power": power,
                "aiclk": aiclk,
                "temp": asic_temperature,
            }

            return chip_telemetry
    
    def get_smbus_board_info(self) -> Dict:
        """Update board info by reading SMBUS_TELEMETRY"""
        pyluwen_chip = self.device
        if pyluwen_chip.as_bh():
            telem_struct = pyluwen_chip.as_bh().get_telemetry()
        elif pyluwen_chip.as_wh():
            telem_struct = pyluwen_chip.as_wh().get_telemetry()
        else:
            telem_struct = pyluwen_chip.as_gs().get_telemetry()
        json_map = jsons.dump(telem_struct)
        smbus_telem_dict = dict.fromkeys(constants.SMBUS_TELEMETRY_LIST)

        for key, value in json_map.items():
            if value:
                smbus_telem_dict[key.upper()] = hex(value)
        return smbus_telem_dict


devices = detect_chips_with_callback(local_only=True)
print("Found devices")
print("Start monitoring..")
telemetry = Telemetry(devices)
telemetry.update_telem()

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

file_name = f"{time.strftime('%m_%d_%H:%M:%S')}.json"
with open (os.path.join(LOG_DIR, file_name), "w") as f:
    f.write("[")
    first = True
    try:
        while(True):
            if first:
                first = False
            else:
                f.write(", ")
            f.write(jsons.dumps(telemetry.device_telemetrys))
            telemetry.update_telem()
            print(telemetry.device_telemetrys)
            time.sleep(1)
    except KeyboardInterrupt:
        f.write("]")
        print("Stop monitoring..")
        print(f"Saved logs in {file_name}")

"""
with open(os.path.join(LOG_DIR, file_name), "r") as f:
    telemetrys_data = jsons.loads(f.read())
    print(telemetrys_data)

df_telemetry = pd.DataFrame.from_dict(telemetrys_data)
"""