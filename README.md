# PredictiveGroceryAI

**AI-powered predictive model to forecast daily grocery consumption and recommend optimal restock dates.**

---

## Overview

This project demonstrates how to use **time series forecasting** to predict grocery consumption and provide smart restocking suggestions. It uses **Facebook Prophet** for modeling and visualization.

> ⚠️ **Note:** In this example, we use **Milk** as the grocery item, but the project can be easily adapted for any grocery item such as eggs, vegetables, snacks, or beverages.

---

## Features

- Predicts daily consumption of a grocery item using historical data.
- Provides a recommended **restock date** based on predicted usage.
- Visualizes actual vs predicted consumption trends.
- Scalable to multiple grocery items by replacing the example dataset.

---

## Installation

Make sure you have Python installed (>=3.8) and then install the required packages:

```bash
pip install pandas numpy matplotlib prophet
