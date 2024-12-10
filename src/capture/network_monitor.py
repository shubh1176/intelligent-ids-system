from scapy.all import sniff
from queue import Queue
import threading

class NetworkMonitor:
    def __init__(self, packet_queue: Queue):
        self.packet_queue = packet_queue
        self.is_running = False
        
    def start_capture(self):
        self.is_running = True
        sniff(prn=self.packet_handler, store=False)
    
    def packet_handler(self, packet):
        if self.is_running:
            print(f"Captured packet: {packet.summary()}")  # Debug print
            self.packet_queue.put(packet) 