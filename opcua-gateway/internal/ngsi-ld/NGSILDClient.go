package ngsildclient

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const MaxSensorValues = 10

type SensorNode struct {
	ID            string
	Type          string
	AttributeName string
	Values        []SensorValue
}

type SensorValue struct {
	Timestamp time.Time
	Value     float64
}

func NormalizeNodeId(nodeId string) string {
	modifiedNodeID := strings.ReplaceAll(nodeId, ";", ",")
	modifiedNodeID = strings.ReplaceAll(modifiedNodeID, "=", ":")
	return modifiedNodeID
}

func (sn SensorNode) ToNGSIld() ([]byte, error) {
	if len(sn.Values) == 0 {
		return nil, fmt.Errorf("sensor node values list is empty")
	}

	// Create a list of readings with timestamps
	readings := make([]map[string]interface{}, 0, len(sn.Values))
	for _, value := range sn.Values {
		reading := map[string]interface{}{
			"type":       "Property",
			"value":      value.Value,
			"observedAt": value.Timestamp.UTC().Format("2006-01-02T15:04:05.000Z"),
		}
		readings = append(readings, reading)
	}

	// Create the NGSI-LD entity structure
	entity := map[string]interface{}{
		"id":   NormalizeNodeId(sn.ID),
		"type": sn.Type,
		sn.AttributeName: map[string]interface{}{
			"type":  "StructuredValue",
			"value": readings,
			"metadata": map[string]interface{}{
				"observedAt": map[string]interface{}{
					"type":  "DateTime",
					"value": sn.Values[len(sn.Values)-1].Timestamp.UTC().Format("2006-01-02T15:04:05.000Z"), // Use latest timestamp
				},
			},
		},
	}

	// Marshal the entity to JSON-LD
	jsonData, err := json.Marshal(entity)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal to JSON-LD: %w", err)
	}

	return jsonData, nil
}

func SendToOrion(orionUrl string, payload []byte) error {
	// Construct the Orion Context Broker URL
	url := fmt.Sprintf("http://%s/v2/entities", orionUrl)

	// Create a new HTTP request
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(payload))
	if err != nil {
		return fmt.Errorf("failed to create HTTP request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	fmt.Println("\nPayload:", string(payload))
	// Send the request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send entity creation request to Orion: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("reading response body failed %w", err)
	}

	// Check the response status code
	if resp.StatusCode != http.StatusCreated {
		return fmt.Errorf("orion returned non-success status code for entity creation: %d %s", resp.StatusCode, string(body))
	}
	return nil
}

func UpdateEntityAttribute(orionUrl string, entityID string, attributeName string, values []SensorValue) error {
	url := fmt.Sprintf("http://%s/v2/entities/%s/attrs/%s/value", orionUrl, entityID, attributeName)

	// Create the list of readings with timestamps for Orion
	readings := make([]map[string]interface{}, 0, len(values))
	for _, value := range values {
		reading := map[string]interface{}{
			"type":       "Property",
			"value":      value.Value,
			"observedAt": value.Timestamp.UTC().Format("2006-01-02T15:04:05.000Z"),
		}
		readings = append(readings, reading)
	}

	// Marshal the readings to JSON
	jsonValue, err := json.Marshal(readings) // Marshal `readings` directly
	if err != nil {
		return fmt.Errorf("failed to marshal new value to JSON: %w", err)
	}

	// Create a new HTTP request
	req, err := http.NewRequest("PUT", url, bytes.NewBuffer(jsonValue))
	if err != nil {
		return fmt.Errorf("failed to create HTTP request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	// Send the request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send request to Orion: %w", err)
	}
	defer resp.Body.Close()

	// Check the response status code
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusNoContent {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("orion returned non-success status code: %d, response body: %s", resp.StatusCode, body)
	}

	return nil
}

func AddValueToSensorNode(node *SensorNode, timestamp time.Time, value float64) {
	newValue := SensorValue{
		Timestamp: timestamp,
		Value:     value,
	}
	node.Values = append(node.Values, newValue)

	if len(node.Values) == MaxSensorValues {
		// Perform actions like sending the data to Orion
		payload, err := node.ToNGSIld()
		if err != nil {
			fmt.Println("Error creating NGSI-LD payload:", err)
			return
		}

		err = SendToOrion("your_orion_url", payload) // Replace with your Orion URL
		if err != nil {
			fmt.Println("Error sending to Orion:", err)
			return
		}

		fmt.Println("Successfully sent data to Orion!")

		node.Values = []SensorValue{}
	}
}
