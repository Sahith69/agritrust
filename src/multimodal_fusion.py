import sys
sys.path.insert(0, '/home/sahith/Projects/Hackathon')
import config

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ─── MULTIMODAL NEURAL NETWORK ARCHITECTURE ──────────────────────────────────
class SensorEncoder(nn.Module):
    """
    Encodes tabular IoT sensor readings:
    - Temperature (°C)
    - Soil Moisture (%)
    - Relative Humidity (%)
    """
    def __init__(self, in_features: int = 3, hidden_dim: int = 32, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if hidden_dim > 1 else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handles single sample or batch
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)


class MultimodalCropFusionNet(nn.Module):
    """
    Joint Multimodal Network:
    Fuses 512-dim CNN visual feature embeddings with 64-dim IoT sensor embeddings
    to produce a combined disease and environmental vulnerability assessment.
    """
    def __init__(self, visual_feature_dim: int = 512, sensor_dim: int = 64, num_classes: int = 4):
        super().__init__()
        self.sensor_encoder = SensorEncoder(in_features=3, hidden_dim=32, out_dim=sensor_dim)
        
        fusion_dim = visual_feature_dim + sensor_dim  # 512 + 64 = 576
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, visual_feats: torch.Tensor, sensor_data: torch.Tensor) -> torch.Tensor:
        # visual_feats: [B, 512], sensor_data: [B, 3]
        if sensor_data.dim() == 1:
            sensor_data = sensor_data.unsqueeze(0)
        if visual_feats.dim() == 1:
            visual_feats = visual_feats.unsqueeze(0)
            
        sensor_feats = self.sensor_encoder(sensor_data)
        joint_representation = torch.cat([visual_feats, sensor_feats], dim=1)
        logits = self.fusion_head(joint_representation)
        return logits


# ─── BIOCLIMATIC AGRONOMIC RULE & RISK ENGINE ────────────────────────────────
def evaluate_environmental_risk(crop: str, disease_name: str, temp_c: float, soil_moisture_pct: float, humidity_pct: float = 75.0):
    """
    Calculates the Bioclimatic Outbreak Risk Index (0 - 100%) based on 
    established plant pathology models (e.g. Mills Period for Apple Scab,
    IRRI blast weather indices for Paddy).
    
    Returns:
        risk_score (float, 0-100)
        risk_level (str: 'Low', 'Moderate', 'High', 'Critical')
        risk_alert (str)
        action_modifier (str)
    """
    score = 25.0
    alerts = []
    
    # 1. PADDY DISEASES
    if crop == "Paddy":
        if "Blast" in disease_name:
            # Blast favors 20-28°C and high moisture/humidity (>80%)
            if 20.0 <= temp_c <= 28.0 and soil_moisture_pct >= 75.0:
                score += 55.0
                alerts.append("Optimal Blast Temperature & Saturated Soil (20-28°C, >75% moisture) accelerates sporulation!")
            elif soil_moisture_pct >= 70.0:
                score += 30.0
        elif "Bacterial Leaf Blight" in disease_name:
            # Blight favors warm (25-34°C) and waterlogged conditions
            if 25.0 <= temp_c <= 34.0 and soil_moisture_pct >= 70.0:
                score += 50.0
                alerts.append("Warm waterlogged environment (25-34°C) promotes rapid bacterial vascular exudation.")
            elif temp_c > 30.0:
                score += 25.0
        else:
            if soil_moisture_pct >= 85.0:
                score += 35.0
                alerts.append("Excess standing water increases fungal root/foliar vulnerability.")

    # 2. APPLE DISEASES
    elif crop == "Apple":
        if "Apple Scab" in disease_name:
            # Mills Period: 15-24°C + wet leaves/soil > 70% moisture = high infection
            if 15.0 <= temp_c <= 24.0 and soil_moisture_pct >= 65.0:
                score += 60.0
                alerts.append("Mills Infection Window Met (15-24°C, High Moisture): Ascomycete spores actively germinate within 6-9 hrs.")
            elif soil_moisture_pct >= 60.0:
                score += 30.0
        elif "Black Rot" in disease_name:
            if 24.0 <= temp_c <= 32.0 and soil_moisture_pct >= 60.0:
                score += 50.0
                alerts.append("Warm humid orchard conditions (24-32°C) trigger pycnidiospore discharge.")
        elif "Cedar Apple Rust" in disease_name:
            if 12.0 <= temp_c <= 24.0 and soil_moisture_pct >= 70.0:
                score += 55.0
                alerts.append("Teliospore germination window active (12-24°C with wet leaf canopy).")

    # 3. ARECANUT DISEASES
    elif crop == "Arecanut":
        if "Mahali" in disease_name or "Koleroga" in disease_name:
            # Phytophthora nut rot spikes during continuous high moisture (>85%) and 22-28°C
            if soil_moisture_pct >= 80.0 and 20.0 <= temp_c <= 29.0:
                score += 65.0
                alerts.append("Severe Monsoon Rot Conditions (>80% Soil Moisture): Phytophthora zoospores actively wash down nut bunches.")
            elif soil_moisture_pct >= 70.0:
                score += 35.0
        elif "Yellow Leaf" in disease_name:
            if soil_moisture_pct < 30.0 and temp_c > 32.0:
                score += 40.0
                alerts.append("Drought stress exacerbates phytoplasma yellowing symptom expression.")

    # General Environmental Stress Modifiers
    if soil_moisture_pct > 85.0:
        score += 15.0
        if "Waterlogged" not in " ".join(alerts):
            alerts.append("Soil is severely waterlogged; anaerobic root stress detected.")
    elif soil_moisture_pct < 25.0:
        score += 10.0
        alerts.append("Low soil moisture; crop experiencing moisture-deficit stress.")

    final_score = min(100.0, max(5.0, score))

    if final_score >= 75.0:
        risk_level = "CRITICAL OUTBREAK RISK"
        action_modifier = "🚨 Immediate spraying required! High environmental disease pressure will cause rapid epidemic spread within 24-48 hours."
    elif final_score >= 50.0:
        risk_level = "MODERATE ELEVATED RISK"
        action_modifier = "⚠️ Environmental conditions favor disease progression. Monitor crop closely and apply preventive fungicide."
    else:
        risk_level = "LOW CLIMATIC PRESSURE"
        action_modifier = "🟢 Weather and soil moisture do not currently favor aggressive disease escalation."

    main_alert = " | ".join(alerts) if alerts else "Soil moisture and temperature are within standard thresholds."
    return final_score, risk_level, main_alert, action_modifier
