package config

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v2"
)

type Config struct {
	OpcuaUrl       string `yaml:"opcua-url"`
	NuclioUrl      string `yaml:"nuclio-url"`
	PredictionPort int    `yaml:"prediction-port"`
	OrionUrl       string `yaml:"orion-url"`
}

func GetConfig(cfg *Config) {
	configpath, exists := os.LookupEnv("CONFIG_PATH")
	if !exists {
		configpath = "../internal/config/config.yml"
	}
	f, err := os.Open(configpath)
	if err != nil {
		processError(err)
	}
	defer f.Close()

	decoder := yaml.NewDecoder(f)
	err = decoder.Decode(cfg)
	if err != nil {
		processError(err)
	}
}

func processError(err error) {
	fmt.Println(err)
	os.Exit(2)
}
