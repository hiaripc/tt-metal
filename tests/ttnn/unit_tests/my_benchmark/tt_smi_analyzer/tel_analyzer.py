from typing import List, Dict
# from pyluwen import PciChip
from . import constants
from tt_tools_common.utils_common.tools_utils import detect_chips_with_callback
from rich.progress import track
import jsons
import time
import os
import pandas as pd
import pathlib
from threading import Thread

LOG_PATH = pathlib.Path(__file__).parent.parent.resolve()

class TelemetryAnalyzer:

    def __init__(self, log_path: str = None):
        
        devices = detect_chips_with_callback(local_only=True)
        print("Found devices")

        # Simplified with just one device (GS)
        self.device = devices[0]        
        self.__update_telem()

        self.monitoring_thread = None
        self.timer_start = None

        self.log_dir = LOG_PATH if not log_path else log_path
        
        self.log_dir = os.path.join(self.log_dir, "tel_analyzer_logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def __update_telem(self):
        """Update telemetry in a given interval"""
        self.smbus_telem_info = self.__get_smbus_board_info()
        self.device_telemetrys = self.__get_wh_gs_chip_telemetry()

 
    def __get_wh_gs_chip_telemetry(self) -> Dict:
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
    
    def __get_smbus_board_info(self) -> Dict:
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


    def start_monitoring_online(self, freq: float = 0.5, verb: bool = False):
        if verb:
            print("Start monitoring..")
        self.__update_telem()

        self.telemetrys_dict = {"voltage": [], "current": [], "power": [], "aiclk": [], "temp": []}
        def monitoring(stop_monitoring, verb):
            try:
                while(not stop_monitoring()):
                    self.__update_telem()
                    for k, v in self.device_telemetrys.items():
                        self.telemetrys_dict[k].append(v)
                    if verb:
                        print(self.device_telemetrys)
                    time.sleep(freq)
            except KeyboardInterrupt:
                pass 
            if verb:
                print("Stop monitoring ...")

        self.timer_start = time.time()
        self._stop_monitoring = False
        self.monitoring_thread = Thread(target=monitoring, args= (lambda: self._stop_monitoring, verb ))
        self.monitoring_thread.start()
        if verb:
            print("Monitoring started")
        return
    
    def stop_monitoring_online(self):
        self._stop_monitoring = True
        return self.telemetrys_dict

    def start_monitoring_logs(self, file_name: str = "", verb: bool = False):
        
        print("Start monitoring..")
        self.__update_telem()

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        def monitoring(file_name, stop_monitoring, verb):
            full_path = os.path.join(self.log_dir, file_name)
            with open (full_path, "w") as f:
                f.write("[")
                first = True
                try:
                    while(not stop_monitoring()):
                        if first:
                            first = False
                        else:
                            f.write(", ")
                        f.write(jsons.dumps(self.device_telemetrys))
                        self.__update_telem()
                        if verb:
                            print(self.device_telemetrys)
                        time.sleep(0.1)
                except KeyboardInterrupt:
                    pass 
                f.write("]")
                print("Stop monitoring ...")
                print(f"Saved logs in {full_path}")

        if file_name:
            file_name = file_name + "_"
        file_name = f"{file_name}{time.strftime('%m%d_%H:%M:%S')}.json"
        self.timer_start = time.time()
        self._stop_monitoring = False
        monitoring_thread = Thread(target=monitoring, args= (file_name, lambda: self._stop_monitoring, verb ))
        monitoring_thread.start()
        print("Monitoring started")
        return        

    def stop_monitoring(self):
        self._stop_monitoring = True
        print("Monitoring stopped")
        if self.timer_start:
            total_time = time.time() - self.timer_start
            print(f"Total time {total_time}")


"""
with open(os.path.join(LOG_DIR, file_name), "r") as f:
    telemetrys_data = jsons.loads(f.read())
    print(telemetrys_data)

df_telemetry = pd.DataFrame.from_dict(telemetrys_data)
"""