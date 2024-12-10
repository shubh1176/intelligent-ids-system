from queue import Queue
import threading
from capture.network_monitor import NetworkMonitor
from dashboard.app import DashboardApp

class MonitorController:
    def __init__(self):
        self.packet_queue = Queue()
        self.network_monitor = NetworkMonitor(self.packet_queue)
        self.dashboard = DashboardApp()
        
    def process_packets(self):
        while True:
            packet = self.packet_queue.get()
            prediction = self.dashboard.process_packet(packet)
            self.dashboard.update_stats(prediction)
            
    def run(self):
        # Start packet processing thread
        processing_thread = threading.Thread(target=self.process_packets)
        processing_thread.daemon = True
        processing_thread.start()
        
        # Start network monitoring
        monitor_thread = threading.Thread(target=self.network_monitor.start_capture)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Run dashboard
        self.dashboard.run() 