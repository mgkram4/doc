# Perfect Pose - Simplified Presentation Diagrams

Clean, simple diagrams perfect for presentation slides with other content.

---

## **System Architecture Overview** 
*Clean system overview for slides*

```mermaid
graph LR
    A[Flutter Mobile App] --> B[Flask API Endpoint]
    B --> C[MediaPipe<br/>Pose Detection]
    C --> D[CNN-LSTM Model<br/>AI Analysis]
    D --> E[Geometric Analysis<br/>Joint Angles]
    E --> A
    A --> F[Firebase Database<br/>User Progress]
    
    style A fill:#2196F3
    style B fill:#4CAF50
    style C fill:#FF9800
    style D fill:#9C27B0
    style E fill:#FFC107
    style F fill:#FF6F00
```

---

## **User Workflow**
*Clean user journey for slides*

```mermaid
flowchart LR
    A[Image Capture] --> B[Pose Detection<br/>33 Landmarks]
    B --> C[Dual Analysis<br/>Geometric + ML]
    C --> D[Specific Feedback<br/>Joint Angles]
    D --> E[Progress Tracking<br/>Firebase Storage]
    
    style A fill:#e3f2fd
    style B fill:#e8f5e8
    style C fill:#fff3e0
    style D fill:#ffebee
    style E fill:#f3e5f5
```

---

## **Technology Stack**
*Key technologies and integrations*

```mermaid
graph TB
    subgraph "Frontend"
        A[Flutter Framework<br/>Cross-Platform Mobile]
        B[Firebase SDK<br/>Authentication & Storage]
    end
    
    subgraph "Backend"
        C[Python Flask API<br/>Pose Analysis Service]
        D[MediaPipe Library<br/>Google Pose Detection]
        E[TensorFlow Models<br/>CNN-LSTM Classification]
    end
    
    A --> C
    A --> B
    C --> D
    C --> E
    
    style A fill:#2196F3
    style B fill:#FF6F00
    style C fill:#4CAF50
    style D fill:#FF9800
    style E fill:#9C27B0
```

---

## **AI Processing Pipeline**
*Detailed analysis workflow*

```mermaid
flowchart LR
    A[Image Input<br/>User Photo] --> B[MediaPipe Processing<br/>Extract 33 Landmarks]
    B --> C[Geometric Analysis<br/>Cosine Distance Calculation]
    B --> D[CNN-LSTM Model<br/>Exercise Classification]
    C --> E[Joint Angle Analysis<br/>8 Key Body Points]
    D --> E
    E --> F[Similarity Scoring<br/>& Feedback Generation]
    
    style A fill:#e1f5fe
    style B fill:#e8f5e8
    style C fill:#fff3e0
    style D fill:#f3e5f5
    style E fill:#ffebee
    style F fill:#f9fbe7
```

---

## **System Components**
*Ultra-simple component view*

```mermaid
graph TD
    A[Mobile App]
    B[AI Backend]
    C[Database]
    
    A --> B
    B --> A
    A --> C
    
    style A fill:#2196F3
    style B fill:#4CAF50
    style C fill:#FF6F00
```

---

## **Exercise Categories**
*Simple category breakdown*

```mermaid
graph LR
    A[Perfect Pose] --> B[Yoga<br/>3 poses]
    A --> C[Weightlifting<br/>3 poses]
    A --> D[Bodyweight<br/>3 poses]
    A --> E[Functional<br/>3 poses]
    
    style A fill:#9C27B0
    style B fill:#4CAF50
    style C fill:#FF9800
    style D fill:#2196F3
    style E fill:#FF6F00
```

---

## **Usage Guide for Slides**

### **Best for Presentation Slides:**
1. **System Architecture Overview** - Shows complete system in 6 components
2. **User Workflow** - User journey in 5 technical steps  
3. **AI Processing Pipeline** - Detailed dual analysis workflow
4. **Technology Stack** - Professional technology overview

### **When to Use:**
- **System Architecture**: During technical solution overview (1:00 mark)
- **User Workflow**: During user experience explanation (2:00 mark)  
- **AI Processing Pipeline**: During detailed technical deep-dive
- **Technology Stack**: For Q&A or technical credibility

### **Why These Work Better:**
- ✅ **Fewer elements** - easier to read on slides
- ✅ **Larger text** - readable from distance
- ✅ **Simple colors** - works with other slide content
- ✅ **Clear flow** - audience can follow easily

### **Slide Layout Tips:**
```
Your presentation text     [Simple diagram]
• Key technical points     [App screenshot]  
• Impressive metrics       
```

### **Color Scheme:**
- 🔵 **Blue**: Frontend/Mobile
- 🟢 **Green**: Backend/API
- 🟠 **Orange**: AI/ML
- 🟡 **Firebase Orange**: Database
- 🟣 **Purple**: Core system 
