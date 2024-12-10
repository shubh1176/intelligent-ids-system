import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import torch
from preprocessing.feature_extractor import FeatureExtractor
from model import ImprovedNetworkModel
import os
import time
from scapy.all import *
import torch.nn.functional as F
import random

class DashboardApp:
    def __init__(self):
        st.set_page_config(layout="wide")
        self.initialize_state()
        self.feature_extractor = FeatureExtractor(model_input_dims=78)
        self.load_model()
        self.traffic_stats = {
            'normal': 0,
            'suspicious': 0,
            'malicious': 0
        }
        self.alerts = []
        
    def initialize_state(self):
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
        if 'traffic_stats' not in st.session_state:
            st.session_state.traffic_stats = {
                'normal': 0,
                'suspicious': 0,
                'malicious': 0
            }
    
    def load_model(self):
        try:
            model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'best_model.pth')
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            self.model = ImprovedNetworkModel(input_dim=78)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            raise
            
    def process_packet(self, packet):
        features = self.feature_extractor.extract(packet)
        with torch.no_grad():
            prediction = self.model(features)
            probabilities = F.softmax(prediction, dim=1)
            result = torch.argmax(prediction).item()
            print(f"Processed packet - Raw prediction: {result}")
            print(f"Probabilities: {probabilities}")
            return result
            
    def update_stats(self, prediction):
        if prediction == 0:
            self.traffic_stats['normal'] += 1
        elif prediction in [1, 2]:
            self.traffic_stats['suspicious'] += 1
            self.add_alert(f"Suspicious traffic detected (Type {prediction})")
        else:
            self.traffic_stats['malicious'] += 1
            self.add_alert(f"Malicious traffic detected! (Type {prediction})", level="danger")
            
    def add_alert(self, message, level="warning"):
        alert = {
            "timestamp": datetime.now(),
            "message": message,
            "level": level
        }
        self.alerts.append(alert)
        
    def render_alerts(self):
        st.subheader("Recent Alerts")
        for alert in reversed(st.session_state.alerts[-10:]):
            color = "red" if alert["level"] == "danger" else "orange"
            st.markdown(
                f"<div style='color:{color}'>{alert['timestamp'].strftime('%H:%M:%S')} - {alert['message']}</div>", 
                unsafe_allow_html=True
            )
            
    def render_traffic_graph(self):
        st.subheader("Traffic Analysis")
        fig = go.Figure()
        stats = st.session_state.traffic_stats
        fig.add_trace(go.Bar(
            x=['Normal', 'Suspicious', 'Malicious'],
            y=[stats['normal'], stats['suspicious'], stats['malicious']],
            marker_color=['green', 'orange', 'red']
        ))
        st.plotly_chart(fig, use_container_width=True)
        
    def simulate_normal_traffic(self):
        from scapy.layers.inet import IP, TCP, UDP
        
        normal_packets = [
            # HTTP traffic
            IP()/TCP(sport=80, dport=random.randint(1024, 65535), flags='A'),
            # HTTPS traffic
            IP()/TCP(sport=443, dport=random.randint(1024, 65535), flags='A'),
            # DNS query
            IP()/UDP(sport=53, dport=random.randint(1024, 65535))
        ]
        
        print("Simulating normal traffic")
        for packet in normal_packets:
            packet.time = time.time()
            packet.proto = 6 if TCP in packet else 17
            prediction = self.process_packet(packet)
            self.update_stats(prediction)

    def simulate_attack_traffic(self):
        from scapy.layers.inet import IP, TCP, UDP
        
        attack_packets = [
            # Port scanning (Type 1)
            IP()/TCP(sport=12345, dport=22, flags='S'),  # SYN scan to SSH
            IP()/TCP(sport=12346, dport=23, flags='S'),  # SYN scan to Telnet
            
            # SYN flood (Type 2)
            IP()/TCP(sport=54321, dport=80, flags='S'*5),  # Multiple SYN flags
            IP()/TCP(sport=54322, dport=443, flags='S'*5), 
            
            # UDP flood (Type 3)
            IP()/UDP(sport=60000, dport=53)/('X'*1000),  # Large UDP packet
            
            # ACK flood (Type 4)
            IP()/TCP(sport=55555, dport=80, flags='FA'*5),  # FIN-ACK flood
            
            # RST flood (Type 5)
            IP()/TCP(sport=44444, dport=8080, flags='R'*3)  # Multiple RST flags
        ]
        
        print("Simulating attack traffic")
        for packet in attack_packets:
            packet.time = time.time()
            packet.proto = 6 if TCP in packet else 17
            prediction = self.process_packet(packet)
            self.update_stats(prediction)

    def render_header(self):
        st.title("Network Security Monitor")
        
        # Add simulation buttons in a row
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            if st.button("Simulate Normal Traffic"):
                self.simulate_normal_traffic()
        with col2:
            if st.button("Simulate Attack", type="primary"):
                self.simulate_attack_traffic()
        with col3:
            st.metric("Active Connections", "127")
        with col4:
            st.metric("Threats Detected", str(self.traffic_stats['malicious']))
        with col5:
            st.metric("Network Load", "1.2 Gbps")
        
    def run(self):
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.traffic_stats = self.traffic_stats
            st.session_state.alerts = self.alerts

        # Render the static header
        self.render_header()
        
        # Create two columns for the main content
        col1, col2 = st.columns([2, 1])
        
        # Create placeholders for dynamic content
        with col1:
            graph_placeholder = st.empty()
        with col2:
            alerts_placeholder = st.empty()
        
        # Update the placeholders every second
        while True:
            with graph_placeholder:
                self.render_traffic_graph()
            with alerts_placeholder:
                self.render_alerts()
            time.sleep(0.5)
            st.experimental_rerun()