import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from utils.logging import setup_logging

logger = setup_logging(__name__)

class Dashboard:
    def __init__(self, root, iteration_data):
        self.root = root
        self.root.title("TradingBot Dashboard")
        self.iteration_data = iteration_data
        
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()
        
        self.update_dashboard()

    def update_dashboard(self):
        self.ax.clear()
        self.ax.plot(self.iteration_data['timestamp'], self.iteration_data['portfolio_value'], label='Portfolio Value')
        self.ax.set_title("Portfolio Value Over Time")
        self.ax.set_xlabel("Timestamp")
        self.ax.set_ylabel("Value ($)")
        self.ax.legend()
        self.canvas.draw()
        self.root.after(1000, self.update_dashboard)