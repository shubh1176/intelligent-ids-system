import streamlit as st
from controller.monitor_controller import MonitorController

def main():
    try:
        controller = MonitorController()
        controller.run()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")

if __name__ == "__main__":
    main() 