package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"

	ngsildclient "opcua-gateway/internal/ngsi-ld"
	"opcua-gateway/internal/opcua"
)

func healthCheck(w http.ResponseWriter, req *http.Request) {
	// Simple health check: respond with 200 OK and a JSON message
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func handleRequests(w http.ResponseWriter, req *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if req.Method != http.MethodPost {
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(req.Body)
	if err != nil {
		http.Error(w, "Error reading request body", http.StatusInternalServerError)
		return
	}
	defer req.Body.Close()

	var requestData []map[string]interface{} // Changed to an array of maps
	if err := json.Unmarshal(body, &requestData); err != nil {
		http.Error(w, "Invalid JSON data", http.StatusBadRequest)
		return
	}

	var sensorDataList []ngsildclient.SensorNode
	for _, sensorData := range requestData {
		// Extract fields from the map and perform type assertions
		id, ok := sensorData["id"].(string)
		if !ok {
			http.Error(w, "Invalid 'id' field", http.StatusBadRequest)
			return
		}

		sensorType, ok := sensorData["type"].(string)
		if !ok {
			http.Error(w, "Invalid 'type' field", http.StatusBadRequest)
			return
		}

		attributeName, ok := sensorData["attributeName"].(string)
		if !ok {
			http.Error(w, "Invalid 'attributeName' field", http.StatusBadRequest)
			return
		}

		// Create SensorData struct
		sensorDataList = append(sensorDataList, ngsildclient.SensorNode{
			ID:            id,
			Type:          sensorType,
			AttributeName: attributeName,
			Values:        []ngsildclient.SensorValue{},
		})
	}

	fmt.Printf("Received sensor data: %+v\n", sensorDataList)

	errSub := opcua.OpcuaHandler(sensorDataList)
	if errSub != nil {
		errorMessage := fmt.Sprintf("Subscription failed: %v", errSub)
		fmt.Printf("Subscription failed: %v\n", errSub)
		http.Error(w, errorMessage, http.StatusInternalServerError) // More appropriate status code
		return
	}

	response := map[string]string{"message": "Subscription started successfully"}
	json.NewEncoder(w).Encode(response)
}

func main() {
	//routes
	http.HandleFunc("/health", healthCheck)
	http.HandleFunc("/start", handleRequests)

	log.Println("API gateway listening on port 8880")
	err := http.ListenAndServe(":8880", nil)
	if err != nil {
		log.Fatalf("Gateway crashed! :(")
		log.Panic(err)
	}
}
