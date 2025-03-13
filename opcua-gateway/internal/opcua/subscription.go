package opcua

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/gopcua/opcua"
	"github.com/gopcua/opcua/ua"

	"opcua-gateway/internal/config"
	ngsildclient "opcua-gateway/internal/ngsi-ld"
	//"opcua-gateway/internal/nuclio"
	//ringbuffer "opcua-gateway/internal/utils"
)

func OpcuaHandler(nodes []ngsildclient.SensorNode) error {
	errChan := make(chan error, len(nodes)) // Buffered channel to hold errors

	for _, node := range nodes {
		go func(node ngsildclient.SensorNode) {
			errChan <- opcuaSub(node, errChan) // Send the return value directly to errChan
		}(node)
	}

	var anyError error
	for range nodes { // Wait for all goroutines to finish
		if err := <-errChan; err != nil {
			anyError = err // Capture the first error encountered
			// You might choose to log the error here or accumulate multiple errors
		}
	}

	return anyError // Return the first error or nil if all succeeded
}

// nodeID: for example "ns=2;i=2"
func opcuaSub(node ngsildclient.SensorNode, errChan chan<- error) error {
	// NGSI-LD entity already created or not
	entityCreated := false

	// configuration
	//rb := ringbuffer.NewRingBuffer(10)
	var cfg config.Config
	config.GetConfig(&cfg)
	var (
		endpoint = cfg.OpcuaUrl
		/*policy   = ""
		mode     = ""
		certFile = ""
		keyFile  = ""*/
		//nodeID   = flag.String("node", "", "node id to subscribe to")
		interval = opcua.DefaultSubscriptionInterval
	)

	ctx := context.Background()

	opts := []opcua.Option{
		/*opcua.SecurityPolicy(policy),
		opcua.SecurityModeString(mode),
		opcua.CertificateFile(certFile),
		opcua.PrivateKeyFile(keyFile),*/
		opcua.AuthAnonymous(),
	}

	log.Printf("Starting client cration...")
	c, err := opcua.NewClient(endpoint, opts...)
	if err != nil {
		return fmt.Errorf("failed to create client: %w", err)
	}
	log.Printf("Client created with URL %s", endpoint)
	if err := c.Connect(ctx); err != nil {
		return fmt.Errorf("failed to connect to client: %w", err)
	}
	log.Printf("Conncected")
	defer c.Close(ctx)

	notifyCh := make(chan *opcua.PublishNotificationData)

	sub, err := c.Subscribe(ctx, &opcua.SubscriptionParameters{
		Interval: interval,
	}, notifyCh)
	if err != nil {
		errChan <- fmt.Errorf("error creating subscription for nodeID %s: %w", node.ID, err)
		return fmt.Errorf("failed to subscribe: %w", err)
	}
	defer sub.Cancel(ctx)
	log.Printf("Created subscription with id %v", sub.SubscriptionID)
	errChan <- nil

	//namespaceURI := "http://examples.freeopcua.github.io"
	//ns, err := c.FindNamespace(ctx, namespaceURI)
	//if err != nil {
	//	log.Fatalf("Error finding namespace: %v", err)
	//}

	//nodeID := fmt.Sprintf("0:Objects/%d:HDC/%d:castingspeed", ns, ns)
	//nodeID := fmt.Sprintf("ns=%d;s=HDC/castingspeed", ns)

	id, err := ua.ParseNodeID(node.ID)
	if err != nil {
		return fmt.Errorf("failed to parse nodeId: %w", err)
	}

	var miCreateRequest *ua.MonitoredItemCreateRequest = valueRequest(id)

	res, err := sub.Monitor(ctx, ua.TimestampsToReturnBoth, miCreateRequest)
	if err != nil || res.Results[0].StatusCode != ua.StatusOK {
		return fmt.Errorf("error monitoring:: %w", err)
	}

	err_mkdir := os.MkdirAll("gateway-logs", os.ModeSticky|os.ModePerm) // 0755 sets permissions
	if err_mkdir != nil {
		log.Fatal(err_mkdir)
	}

	fileName := fmt.Sprintf("gateway-logs/logs-%s.log", node.ID)

	batchCount := 0

	file, err := os.OpenFile(fileName, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Println("Error opening file:", err)
	}
	defer file.Close()

	// read from subscription's notification channel until ctx is cancelled
	for {
		select {
		case <-ctx.Done():
			return nil
		case res := <-notifyCh:
			if res.Error != nil {
				log.Print(res.Error)
				continue
			}

			switch x := res.Value.(type) {
			case *ua.DataChangeNotification:
				for _, item := range x.MonitoredItems {
					arrivalTime := time.Now().Format("2006-01-02 15:04:05.000")
					data := item.Value.Value.Value()
					log.Printf("MonitoredItem with client handle %v = %v", item.ClientHandle, data)
					var dataFloat float32

					switch v := data.(type) {
					case float32:
						dataFloat = v
					case float64:
						dataFloat = float32(v)
					case int16:
						dataFloat = float32(v)
					default:
						// Handle the case where 'data' is not a float32 or float64
						log.Printf("Error: Expected float32 or float64, got %T", data)
						continue // Skip to the next item in the loop
					}

					/*
						timestamp := time.Now().Format("2006-01-02 15:04:05.000")
						line := fmt.Sprintf("%s: %f\n", timestamp, dataFloat)
						_, err = file.WriteString(line)
						if err != nil {
							fmt.Println("Error writing to file:", err)
						}
					*/

					newValue := ngsildclient.SensorValue{
						Timestamp: time.Now(),
						Value:     float64(dataFloat),
					}
					node.Values = append(node.Values, newValue)
					if len(node.Values) == 10 {
						if !entityCreated {
							payload, err := node.ToNGSIld()
							if err != nil {
								fmt.Println("Error mapping to NGSI-LD format:", err)
							}
							orion_err := ngsildclient.SendToOrion(cfg.OrionUrl, payload)
							if orion_err != nil {
								if strings.Contains(orion_err.Error(), "Already Exists") {
									err := ngsildclient.UpdateEntityAttribute(
										cfg.OrionUrl, ngsildclient.NormalizeNodeId(node.ID), node.AttributeName, node.Values)
									if err != nil {
										log.Fatal(err)
									}
								} else {
									fmt.Println("Sending to orion failed:", orion_err)
								}
								entityCreated = true
							}
						} else {
							err := ngsildclient.UpdateEntityAttribute(
								cfg.OrionUrl, ngsildclient.NormalizeNodeId(node.ID), node.AttributeName, node.Values)
							if err != nil {
								log.Fatal(err)
							}
						}
						jsonString, err := json.Marshal(node.Values)
						if err != nil {
							// Handle error
						}

						endTime := time.Now().Format("2006-01-02 15:04:05.000")

						line := fmt.Sprintf("Batch %d, Arrival Time: %s, End Time: %s, Value List: %s\n",
							batchCount,
							arrivalTime,
							endTime,
							jsonString)

						_, err = file.WriteString(line)
						if err != nil {
							fmt.Println("Error writing to file:", err)
						}
						node.Values = []ngsildclient.SensorValue{}
						batchCount = batchCount + 1

					}
				}
			default:
				log.Printf("what's this publish result? %T", res.Value)
			}
		}
	}
}

func valueRequest(nodeID *ua.NodeID) *ua.MonitoredItemCreateRequest {
	handle := uint32(42)
	return opcua.NewMonitoredItemCreateRequestWithDefaults(nodeID, ua.AttributeIDValue, handle)
}
