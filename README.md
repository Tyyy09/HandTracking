Real‑Time Hand & Object Detection with LLM Narration
Overview
This project blends computer vision and language reasoning to create a real‑time scene‑understanding system. Using your webcam, it detects objects with YOLOv8, tracks hands with MediaPipe, and sends that information to a Groq LLM to generate a short natural‑language description of what’s happening in the scene.

What It Does
The application continuously analyzes each frame from your camera and overlays:

Color‑coded bounding boxes for detected objects

Hand center markers for up to two hands

A wrapped LLM‑generated caption describing the scene

Smooth, real‑time performance in a resizable OpenCV window

The result is an interactive AI pipeline that narrates real‑world actions as they happen.

Why It’s Interesting
This project demonstrates how vision models and language models can work together to interpret live video. It’s a compact example of multimodal AI — combining detection, tracking, and reasoning into one cohesive system.
