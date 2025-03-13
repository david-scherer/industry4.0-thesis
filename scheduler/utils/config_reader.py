import configparser
import os


class ConfigReader:
    def __init__(self, config_path=None):
        self.config = configparser.ConfigParser()
        if config_path is None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            config_path = os.path.join(dir_path, "..", "config", "config.ini")
        self.config_path = config_path
        self._load_config()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file {self.config_path} does not exist!")
        self.config.read(self.config_path)

    def read_config(self, section, key):
        try:
            return self.config[section][key]
        except KeyError as e:
            raise KeyError(
                f"Key '{key}' not found in section '{section}' of config file!"
            ) from e

    def section_exists(self, section):
        return self.config.has_section(section)