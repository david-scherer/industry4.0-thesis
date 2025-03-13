import asyncio

from asyncua import Client, Node, ua
import os

url = os.getenv('OPCUA_SERVER', "opc.tcp://0.0.0.0:4840/freeopcua/server/")

namespace = "http://examples.freeopcua.github.io"


class SubscriptionHandler:
    """
    The SubscriptionHandler is used to handle the data that is received for the subscription.
    """
    def datachange_notification(self, node: Node, val, data):
        """
        Callback for asyncua Subscription.
        This method will be called when the Client received a data change message from the Server.
        """
        print(f'datachange_notification {node} {val}')


async def main_sub():
    """
    Main task of this Client-Subscription example.
    """
    client = Client(url=url)
    async with client:
        nsidx = await client.get_namespace_index(namespace)
        var = await client.nodes.root.get_child(
            f"0:Objects/{nsidx}:HDC/{nsidx}:castingspeed"
        )
        print(f"0:Objects/{nsidx}:HDC/{nsidx}:castingspeed")
        print(var)
        handler = SubscriptionHandler()
        # We create a Client Subscription.
        subscription = await client.create_subscription(500, handler)
        nodes = [
            var,
            client.get_node(ua.ObjectIds.Server_ServerStatus_CurrentTime),
        ]
        # We subscribe to data changes for two nodes (variables).
        await subscription.subscribe_data_change(nodes)
        # We let the subscription run for ten seconds
        await asyncio.sleep(1)
        # We delete the subscription (this un-subscribes from the data changes of the two variables).
        # This is optional since closing the connection will also delete all subscriptions.
        await subscription.delete()
        # After one second we exit the Client context manager - this will close the connection.
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main_sub())