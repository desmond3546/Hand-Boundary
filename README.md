# Hand-Boundary
This project is a real-time computer-vision prototype that detects a user’s hand in a camera feed and determines how close it is to a virtual on-screen boundary. When the hand crosses the boundary, the system displays a strong warning:
DANGER DANGER
This POC is implemented without MediaPipe, OpenPose, or cloud AI, using only OpenCV + NumPy and classical CV techniques.
Features
Real-time hand/fingertip tracking (classical CV only)
Uses:
HSV skin segmentation
Morphology cleanup
Largest contour selection
Convex hull visualization
Centroid estimation
Fingertip approximation using max-distance heuristic

Virtual boundary (on-screen box)
The system draws a fixed virtual object (rectangle).
The hand’s distance from this box determines the interaction state.
Distance-based interaction states
The prototype continuously measures distance from hand → box boundary and classifies into:
| State       | Meaning                                      |
| ----------- | -------------------------------------------- |
| **SAFE**    | Hand far from boundary                       |
| **WARNING** | Hand approaching the virtual object          |
| **DANGER**  | Hand extremely close / touching the boundary |

Visual overlays
The live camera feed shows:
Current state (SAFE / WARNING / DANGER)
Big red “DANGER DANGER” text when triggered
Hand contour + convex hull in green
Fingertip (red) + selected tracking point (yellow)
FPS counter
Mask preview (bottom-right)
