# Google Analytics Tracking Workflow

## System Overview

This document shows how Google Analytics 4 (GA4) tracks events across the entire ClearVocals platform.

---

## Environment Variables

```env
GA_MEASUREMENT_ID=G-WHZ0X8J9PJ
GA_API_SECRET=ePmKGNiGS126gxaqp6N-4w
ENABLE_ANALYTICS=true
```

---

## Architecture Diagram

```mermaid
graph TB
    subgraph "User Actions"
        U1[User Uploads Video]
        U2[User Purchases Credits]
        U3[User Activates Subscription]
        U4[User Uses Feature]
    end

    subgraph "Frontend - Next.js"
        F1[layout.tsx<br/>GA4 Script Loaded]
        F2[analytics.ts<br/>Tracking Utility]
        
        F3[Payment Success Page]
        F4[Subscription Success Page]
        F5[Dubbing Start Page]
        F6[Clips Upload Component]
        F7[Separation Hook]
        
        F2 --> F3
        F2 --> F4
        F2 --> F5
        F2 --> F6
        F2 --> F7
    end

    subgraph "Backend - FastAPI"
        B1[analytics_service.py<br/>Tracking Service]
        
        B2[elevenlabs_service.py<br/>Text-to-Speech]
        B3[fish_speech_service.py<br/>Voice Generation]
        B4[clip_service.py<br/>GPT-4 Processing]
        B5[ai_segmentation_service.py<br/>Translation/Dubbing]
        
        B1 --> B2
        B1 --> B3
        B1 --> B4
        B1 --> B5
    end

    subgraph "Google Analytics 4"
        GA1[GA4 Real-time<br/>Events Stream]
        GA2[Event Processing]
        GA3[Reports & Dashboards]
        GA4[BigQuery Export<br/>Optional]
    end

    U1 --> F5
    U1 --> F6
    U1 --> F7
    U2 --> F3
    U3 --> F4
    U4 --> F5
    U4 --> F6
    U4 --> F7

    F1 -.->|gtag.js| GA1
    F3 -->|analytics.purchase| GA1
    F4 -->|analytics.subscribe| GA1
    F5 -->|analytics.featureUse| GA1
    F6 -->|analytics.featureUse| GA1
    F7 -->|analytics.featureUse| GA1

    B2 -->|track_api_usage| B1
    B3 -->|track_api_usage| B1
    B4 -->|track_api_usage| B1
    B5 -->|track_api_usage| B1
    B1 -->|Measurement Protocol| GA2

    GA1 --> GA2
    GA2 --> GA3
    GA2 -.-> GA4

    style F1 fill:#4285f4,color:#fff
    style F2 fill:#4285f4,color:#fff
    style B1 fill:#34a853,color:#fff
    style GA2 fill:#fbbc04,color:#000
    style GA3 fill:#ea4335,color:#fff
```

---

## Event Flow Diagrams

### 1. Purchase Flow

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant Stripe
    participant Backend
    participant GA4

    User->>Frontend: Click "Buy Credits"
    Frontend->>Stripe: Redirect to Checkout
    Stripe->>Frontend: Return to success page
    Frontend->>Backend: Verify Payment
    Backend-->>Frontend: Payment Confirmed
    Frontend->>GA4: analytics.purchase()<br/>{transaction_id, value, items}
    GA4-->>Frontend: Event Recorded (204)
    
    Note over GA4: Event appears in<br/>Realtime & Reports
```

### 2. Feature Usage Flow (Dubbing)

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant Backend
    participant ElevenLabs
    participant GA4

    User->>Frontend: Upload Video + Start Dubbing
    Frontend->>GA4: analytics.featureUse('dubbing')
    Frontend->>Backend: POST /dubbing/start
    Backend->>ElevenLabs: Generate Speech
    
    Note over Backend: Track API call<br/>start time
    
    ElevenLabs-->>Backend: Audio Response
    
    Note over Backend: Calculate duration<br/>& character count
    
    Backend->>GA4: track_api_usage()<br/>{provider: elevenlabs,<br/>chars: 1234,<br/>success: true}
    
    Backend-->>Frontend: Dubbing Complete
    
    Note over GA4: Both events visible<br/>in Analytics
```

### 3. API Usage Tracking (Backend)

```mermaid
sequenceDiagram
    participant Service as FastAPI Service
    participant Analytics as analytics_service.py
    participant GA4 as Google Analytics 4

    Service->>Service: start_time = time.time()
    Service->>External: Call API<br/>(ElevenLabs/FishSpeech/OpenAI)
    
    alt API Success
        External-->>Service: Response 200
        Service->>Service: success = True
    else API Failure
        External-->>Service: Error
        Service->>Service: success = False
    end
    
    Service->>Service: duration = time.time() - start_time
    Service->>Analytics: track_api_usage()<br/>{provider, chars/tokens,<br/>duration, success}
    
    Note over Analytics: Async fire-and-forget<br/>No blocking
    
    Analytics->>GA4: POST Measurement Protocol<br/>{client_id, event, params}
    GA4-->>Analytics: 204 No Content
    
    Note over Service: Continue processing<br/>(non-blocking)
```

---

## Event Types & Data Structure

### Frontend Events

#### 1. Purchase Event
```mermaid
graph LR
    A[purchase] --> B[transaction_id]
    A --> C[value: USD]
    A --> D[currency: USD]
    A --> E[items array]
    E --> F[item_id: plan]
    E --> G[item_name]
    E --> H[price]
    E --> I[quantity: 1]
```

**Trigger:** Payment success page  
**File:** `frontend/src/app/payment/success/page.tsx`

#### 2. Subscribe Event
```mermaid
graph LR
    A[subscribe] --> B[subscription_type]
    B --> C[pay as you go]
    B --> D[premium]
```

**Trigger:** Subscription activation  
**File:** `frontend/src/app/subscription/success/page.tsx`

#### 3. Feature Use Event
```mermaid
graph LR
    A[feature_use] --> B[feature]
    B --> C[dubbing]
    B --> D[clips]
    B --> E[separation]
```

**Trigger:** User starts feature  
**Files:** 
- `frontend/src/app/workspace/dubbing/start/page.tsx`
- `frontend/src/components/clips/ClipUpload.tsx`
- `frontend/src/hooks/useSeparation.ts`

---

### Backend Events

#### 4. API Usage Event
```mermaid
graph LR
    A[api_usage] --> B[provider]
    A --> C[user_id]
    A --> D[chars/tokens]
    A --> E[success]
    A --> F[timestamp]
    
    B --> G[elevenlabs]
    B --> H[fishspeech]
    B --> I[openai]
```

**Trigger:** Third-party API call  
**Files:**
- `backend/app/services/dub/elevenlabs_service.py`
- `backend/app/services/dub/fish_speech_service.py`
- `backend/app/services/clip_service.py`
- `backend/app/services/dub/ai_segmentation_service.py`

---

## Data Flow Summary

```mermaid
flowchart TD
    Start([User Action]) --> Decision{Action Type?}
    
    Decision -->|Purchase| FE1[Frontend Tracking]
    Decision -->|Feature Use| FE2[Frontend + Backend Tracking]
    Decision -->|Subscription| FE3[Frontend Tracking]
    
    FE1 --> GA1[GA4 via gtag.js]
    FE2 --> GA1
    FE2 --> Backend[Backend API Call]
    FE3 --> GA1
    
    Backend --> ThirdParty{Third-party<br/>API Call?}
    ThirdParty -->|Yes| Track[Track API Usage]
    ThirdParty -->|No| Skip[Skip Tracking]
    
    Track --> GA2[GA4 via Measurement Protocol]
    
    GA1 --> Process[Event Processing]
    GA2 --> Process
    
    Process --> Reports[Reports & Dashboards]
    Reports --> End([Business Insights])
    
    style Start fill:#4285f4,color:#fff
    style End fill:#34a853,color:#fff
    style GA1 fill:#fbbc04,color:#000
    style GA2 fill:#fbbc04,color:#000
    style Reports fill:#ea4335,color:#fff
```

---

## Tracking Coverage

### ✅ Fully Tracked

| Action | Event | Location | Status |
|--------|-------|----------|--------|
| Credit Purchase | `purchase` | Frontend | ✅ |
| Subscription Activation | `subscribe` | Frontend | ✅ |
| Start Dubbing | `feature_use` | Frontend | ✅ |
| Start Clips | `feature_use` | Frontend | ✅ |
| Start Separation | `feature_use` | Frontend | ✅ |
| ElevenLabs API Call | `api_usage` | Backend | ✅ |
| FishSpeech API Call | `api_usage` | Backend | ✅ |
| OpenAI GPT Call | `api_usage` | Backend | ✅ |

---

## View Your Data

### Real-time Dashboard
```mermaid
graph LR
    A[Google Analytics] --> B[Reports]
    B --> C[Realtime]
    C --> D[See Events Live]
    
    style A fill:#4285f4,color:#fff
    style D fill:#34a853,color:#fff
```

**URL:** https://analytics.google.com → Reports → Realtime

### Custom Reports
```mermaid
graph TB
    A[Explore] --> B[Create Report]
    B --> C1[API Usage by Provider]
    B --> C2[Revenue by Feature]
    B --> C3[User Journey Funnel]
    B --> C4[API Performance Monitor]
    
    style A fill:#4285f4,color:#fff
    style C1 fill:#34a853,color:#fff
    style C2 fill:#34a853,color:#fff
    style C3 fill:#34a853,color:#fff
    style C4 fill:#34a853,color:#fff
```

---

## Technical Implementation

### Frontend Stack
```mermaid
graph LR
    A[Next.js App] --> B[layout.tsx]
    B --> C[gtag.js loaded]
    C --> D[utils/analytics.ts]
    D --> E[Components call analytics.*]
    E --> F[Events sent to GA4]
    
    style A fill:#000,color:#fff
    style F fill:#fbbc04,color:#000
```

### Backend Stack
```mermaid
graph LR
    A[FastAPI] --> B[analytics_service.py]
    B --> C[asyncio.create_task]
    C --> D[httpx.AsyncClient]
    D --> E[POST to Measurement Protocol]
    E --> F[GA4 receives event]
    
    style A fill:#009688,color:#fff
    style F fill:#fbbc04,color:#000
```

---

## Cost Tracking (Future Enhancement)

```mermaid
flowchart TD
    A[API Usage Event] --> B{Provider?}
    
    B -->|ElevenLabs| C1[chars × $0.0001]
    B -->|FishSpeech| C2[Custom pricing]
    B -->|OpenAI| C3[tokens × $0.002]
    
    C1 --> D[Estimated Cost]
    C2 --> D
    C3 --> D
    
    D --> E[GA4 Custom Metric]
    E --> F[Cost Reports]
    
    style A fill:#4285f4,color:#fff
    style D fill:#34a853,color:#fff
    style F fill:#ea4335,color:#fff
```

---

## Success Metrics

### KPIs You Can Track

```mermaid
mindmap
  root((Analytics KPIs))
    Revenue
      Total Revenue
      Revenue by Plan
      ARPU
      Conversion Rate
    Usage
      Active Users
      Feature Popularity
      Session Duration
    API Costs
      Total API Calls
      Cost by Provider
      Cost per User
    Performance
      API Success Rate
      Average Response Time
      Error Rate
```

---

## Getting Started

1. **✅ Environment Variables Set**
   ```env
   GA_MEASUREMENT_ID=G-WHZ0X8J9PJ
   GA_API_SECRET=ePmKGNiGS126gxaqp6N-4w
   ENABLE_ANALYTICS=true
   ```

2. **✅ Frontend Tracking Active**
   - Purchase events
   - Subscription events
   - Feature usage events

3. **✅ Backend Tracking Active**
   - ElevenLabs API calls
   - FishSpeech API calls
   - OpenAI API calls

4. **📊 View Data**
   - Real-time: https://analytics.google.com → Realtime
   - Reports: https://analytics.google.com → Reports
   - Custom: https://analytics.google.com → Explore

---

## Troubleshooting Flow

```mermaid
flowchart TD
    Start([Events not showing?]) --> Check1{API Secret set?}
    
    Check1 -->|No| Fix1[Add GA_API_SECRET to .env]
    Check1 -->|Yes| Check2{Check logs?}
    
    Check2 --> Check3{Errors in logs?}
    Check3 -->|Yes| Fix2[Fix API errors]
    Check3 -->|No| Check4{Wait 24 hours?}
    
    Check4 -->|No| Fix3[Wait for GA4 processing]
    Check4 -->|Yes| Check5{Check Realtime?}
    
    Check5 -->|Events there| Success1[✅ Working - wait for reports]
    Check5 -->|No events| Fix4[Check network requests]
    
    Fix1 --> Restart[Restart Backend]
    Fix2 --> Restart
    Restart --> Test[Test API call]
    Test --> Success2[✅ Events appear]
    
    style Start fill:#ea4335,color:#fff
    style Success1 fill:#34a853,color:#fff
    style Success2 fill:#34a853,color:#fff
```

---

**সম্পূর্ণ Analytics System এখন Active! 🎯📊**

