package nuclio

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
)

// DataRequest represents the structure of the data to be sent to the Nuclio function
type DataRequest struct {
	SensorID    string    `json:"sensor_id"`
	SensorValue []float32 `json:"sensor_values"`
}

func SendNuclioMessage(host string, port int, dataList []float32) error { // Return an error to indicate success or failure
	// 1. Define Nuclio Function Endpoint
	nuclioEndpoint := host + ":" + strconv.Itoa(port)

	// 2. Create Sample Sensor Data
	sensorData := DataRequest{
		SensorID:    "sensor123",
		SensorValue: dataList,
	}

	// 3. Prepare Request Payload
	payload, err := json.Marshal(sensorData)
	if err != nil {
		return fmt.Errorf("error marshalling data: %w", err) // Wrap the error for better context
	}

	// 4. Make HTTP Request to Nuclio
	response, err := http.Post(nuclioEndpoint, "application/json", bytes.NewBuffer(payload))
	if err != nil {
		return fmt.Errorf("error making HTTP request: %w", err)
	}
	defer response.Body.Close()

	if response.StatusCode != http.StatusOK {
		// Handle non-OK status codes as errors
		return fmt.Errorf("unexpected Nuclio response status: %s", response.Status)
	}

	bodyBytes, err := io.ReadAll(response.Body)
	if err != nil {
		return fmt.Errorf("error reading response body: %w", err)
	}
	bodyString := string(bodyBytes)
	fmt.Println("Nuclio Response Body:", bodyString)

	// 5. Handle Nuclio Response (Optional)
	fmt.Println("Nuclio Response Status:", response.Status)

	return nil // Indicate success
}
