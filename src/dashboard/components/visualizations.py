class NetworkVisualizations:
    def create_traffic_chart(self, data):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['timestamp'],
            y=data['packets'],
            mode='lines',
            name='Network Traffic'
        ))
        fig.update_layout(
            title='Real-time Network Traffic',
            xaxis_title='Time',
            yaxis_title='Packets/s',
            template='plotly_dark'
        )
        return fig
    
    def create_threat_map(self, threats):
        # World map showing threat origins
        fig = go.Figure(data=go.Scattergeo(
            lon=threats['longitude'],
            lat=threats['latitude'],
            mode='markers',
            marker=dict(
                size=8,
                color='red',
                symbol='circle'
            )
        ))
        return fig 