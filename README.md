# HS-TasNet: Real-Time Low-Latency Music Source Separation

### **Authors**
Satvik Venkatesh, Arthur Benilov, Philip Coleman, Frederic Roskam  
*L-Acoustics, London, N65EG*

---

## **Abstract**
HS-TasNet (Hybrid Spectrogram Time-domain Audio Separation Network) is a real-time low-latency music source separation model. It leverages both spectral and waveform domains for efficient demixing of vocals, drums, bass, and other sources.  
Key highlights include:
- **Low latency**: 23 ms frame size for real-time performance.
- **Competitive SDR**: Achieved 4.65 on MusDB test set, and up to 5.55 with additional training data.
- **Applications**: Live audio remixing, hearing aids, and live stage performances.

---

## **Features**
- **Hybrid Architecture**: Combines spectrogram-based and waveform-based processing for optimal performance.
- **Real-Time Processing**: Adapts advanced models like X-UMX and TasNet for low-latency operation.
- **Flexible Design**: Two variants available:
  - **HS-TasNet**: Full-feature model.
  - **HS-TasNet-Small**: Optimized for reduced computational cost.

---

## **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/HS-TasNet.git
   cd HS-TasNet
