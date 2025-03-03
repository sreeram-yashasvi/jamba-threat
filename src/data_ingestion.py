import asyncio
from azure.eventhub.aio import EventHubConsumerClient
from azure.storage.blob import BlobServiceClient
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThreatFeedIngestion:
    def __init__(self, eventhub_conn_str, eventhub_name, storage_conn_str, container_name):
        self.eventhub_conn_str = eventhub_conn_str
        self.eventhub_name = eventhub_name
        self.blob_service_client = BlobServiceClient.from_connection_string(storage_conn_str)
        self.container_name = container_name

    async def process_event(self, event):
        """Process each event from the threat feed."""
        try:
            event_data = json.loads(event.body_as_str())
            
            # Create a unique filename based on timestamp
            blob_name = f"threat_data/{event.enqueued_time:%Y/%m/%d/%H_%M_%S}_{event.sequence_number}.json"
            
            # Upload to blob storage
            container_client = self.blob_service_client.get_container_client(self.container_name)
            container_client.upload_blob(
                name=blob_name,
                data=json.dumps(event_data),
                overwrite=True
            )
            
            logger.info(f"Successfully processed and stored event: {blob_name}")
            
        except Exception as e:
            logger.error(f"Error processing event: {str(e)}")

    async def receive_events(self):
        """Receive events from Event Hub."""
        client = EventHubConsumerClient.from_connection_string(
            conn_str=self.eventhub_conn_str,
            consumer_group="$Default",
            eventhub_name=self.eventhub_name
        )
        
        async def on_event(partition_context, event):
            await self.process_event(event)
            await partition_context.update_checkpoint(event)

        try:
            await client.receive(on_event=on_event, starting_position="-1")
        except Exception as e:
            logger.error(f"Error receiving events: {str(e)}")
        finally:
            await client.close()

if __name__ == "__main__":
    # Replace with your actual connection strings and names
    EVENTHUB_CONN_STR = "your_eventhub_connection_string"
    EVENTHUB_NAME = "your_eventhub_name"
    STORAGE_CONN_STR = "your_storage_connection_string"
    CONTAINER_NAME = "threat-data"

    ingestion = ThreatFeedIngestion(
        EVENTHUB_CONN_STR,
        EVENTHUB_NAME,
        STORAGE_CONN_STR,
        CONTAINER_NAME
    )
    
    asyncio.run(ingestion.receive_events()) 