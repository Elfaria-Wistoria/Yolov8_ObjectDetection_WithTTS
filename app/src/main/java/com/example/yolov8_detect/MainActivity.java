package com.example.yolov8_detect;

import android.annotation.SuppressLint;
import android.graphics.Bitmap;
import android.media.Image;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.WindowManager;

import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;

import com.google.common.util.concurrent.ListenableFuture;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Locale;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class MainActivity extends AppCompatActivity {
    private ProcessCameraProvider processCameraProvider;
    private PreviewView previewView;
    private RectView rectView;
    private SupportOnnx supportOnnx;
    private OrtEnvironment ortEnvironment;
    private OrtSession ortSession;
    private TextToSpeech textToSpeech;
    private long lastSpeakTime = 0;
    private static final long SPEAK_DELAY = 3000; // Delay 3 detik antara pengucapan

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Inisialisasi views
        previewView = findViewById(R.id.previewView);
        rectView = findViewById(R.id.rectView);

        // Inisialisasi Text-to-Speech
        initializeTextToSpeech();

        // Mencegah layar mati
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        // Cek permission kamera
        permissionCheck();

        // Inisialisasi ONNX Support
        supportOnnx = new SupportOnnx(this);

        // Load model dan memulai kamera
        load();
        setCamera();
        startCamera();
    }

    private void initializeTextToSpeech() {
        textToSpeech = new TextToSpeech(this, status -> {
            if (status == TextToSpeech.SUCCESS) {
                int result = textToSpeech.setLanguage(new Locale("id", "ID")); // Set Bahasa Indonesia

                if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    Log.e("TTS", "Bahasa tidak didukung");
                } else {
                    // Set properti tambahan untuk TTS
                    textToSpeech.setPitch(1.0f);    // Pitch normal
                    textToSpeech.setSpeechRate(1.0f); // Kecepatan normal
                }
            } else {
                Log.e("TTS", "Inisialisasi Text-to-Speech gagal");
            }
        });
    }

    public void permissionCheck() {
        PermissionSupport permissionSupport = new PermissionSupport(this, this);
        permissionSupport.checkPermissions();
    }

    public void load() {
        // Load model dan label
        supportOnnx.loadModel();
        supportOnnx.loadLabel();

        try {
            // Inisialisasi ONNX Runtime
            ortEnvironment = OrtEnvironment.getEnvironment();
            ortSession = ortEnvironment.createSession(
                    this.getFilesDir().getAbsolutePath() + "/" + SupportOnnx.fileName,
                    new OrtSession.SessionOptions()
            );
        } catch (OrtException e) {
            Log.e("ONNX", "Error loading model: " + e.getMessage());
            e.printStackTrace();
        }
    }

    public void setCamera() {
        try {
            ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                    ProcessCameraProvider.getInstance(this);
            processCameraProvider = cameraProviderFuture.get();
        } catch (ExecutionException | InterruptedException e) {
            Log.e("Camera", "Error setting up camera: " + e.getMessage());
            e.printStackTrace();
        }
    }

    public void startCamera() {
        // Set preview view properties
        previewView.setScaleType(PreviewView.ScaleType.FILL_CENTER);

        // Configure camera selector (using back camera)
        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();

        // Build preview use case
        Preview preview = new Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                .build();

        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        // Build image analysis use case
        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();

        // Set labels untuk RectView
        rectView.setLabels(supportOnnx.getLabels());

        // Set image analysis analyzer
        imageAnalysis.setAnalyzer(Executors.newSingleThreadExecutor(), imageProxy -> {
            imageProcessing(imageProxy);
            imageProxy.close();
        });

        // Bind use cases to lifecycle
        processCameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalysis
        );
    }

    @SuppressLint("UnsafeOptInUsageError")
    public void imageProcessing(ImageProxy imageProxy) {
        Image image = imageProxy.getImage();
        if (image != null) {
            try {
                // Konversi image ke bitmap
                Bitmap bitmap = supportOnnx.imageToBitmap(image);
                Bitmap bitmap_640 = supportOnnx.rescaleBitmap(bitmap);

                // Konversi bitmap ke float buffer
                FloatBuffer imgDataFloat = supportOnnx.bitmapToFloatBuffer(bitmap_640);

                // Get input name dan create tensor
                String inputName = ortSession.getInputNames().iterator().next();
                long[] shape = {
                        SupportOnnx.BATCH_SIZE,
                        SupportOnnx.PIXEL_SIZE,
                        SupportOnnx.INPUT_SIZE,
                        SupportOnnx.INPUT_SIZE
                };

                // Create tensor dan run inference
                OnnxTensor inputTensor = OnnxTensor.createTensor(ortEnvironment, imgDataFloat, shape);
                OrtSession.Result result = ortSession.run(Collections.singletonMap(inputName, inputTensor));
                float[][][] output = (float[][][]) result.get(0).getValue();

                // Process results
                int rows = output[0][0].length;
                ArrayList<Result> results = supportOnnx.outputsToNMSPredictions(output, rows);

                // Generate audio feedback
                speakDetectionResults(results);

                // Update visual feedback
                results = rectView.transFormRect(results);
                rectView.clear();
                rectView.resultToList(results);
                rectView.invalidate();

            } catch (OrtException e) {
                Log.e("Inference", "Error during inference: " + e.getMessage());
                e.printStackTrace();
            }
        }
    }

    private void speakDetectionResults(ArrayList<Result> results) {
        long currentTime = System.currentTimeMillis();
        if (currentTime - lastSpeakTime < SPEAK_DELAY) {
            return;
        }

        StringBuilder detectionText = new StringBuilder();
        boolean hasDetections = false;

        for (Result result : results) {
            String label = supportOnnx.getLabels()[result.getLabel()];
            float confidence = result.getScore() * 100;

            if (confidence > 50) { //nilai confidence 45, artinya object dengan akurasi baca 45% baru bisa ada suara
                hasDetections = true;

                // Customized announcements based on object type
                switch(label) {
                    case "ZebraCross":
                        detectionText.append("Zebra cross di depan, ");
                        break;
                    case "RambuBelokKanan":
                        detectionText.append("Rambu belok kanan di depan, ");
                        break;
                    case "RambuBelokKiri":
                        detectionText.append("Rambu belok kiri di depan, ");
                        break;
                    case "RambuPutarBalik":
                        detectionText.append("Rambu putar balik di depan, ");
                        break;
                    case "RambuTitikKumpul":
                        detectionText.append("Titik kumpul di depan, ");
                        break;
                    case "GedungFTMM":
                        detectionText.append("Gedung FTMM terdeteksi, ");
                        break;
                    case "orang":
                        detectionText.append("Ada orang di depan, ");
                        break;
                    case "Mobil":
                        detectionText.append("Ada mobil di depan, ");
                        break;
                    case "sepedamotor":
                        detectionText.append("Ada sepeda motor di depan, ");
                        break;
                }
            }
        }

        if (hasDetections) {
            String textToSpeak = detectionText.toString().trim();
            if (textToSpeak.endsWith(",")) {
                textToSpeak = textToSpeak.substring(0, textToSpeak.length() - 1);
            }

            textToSpeech.speak(textToSpeak, TextToSpeech.QUEUE_FLUSH, null, null);
            lastSpeakTime = currentTime;
        }
    }

    @Override
    protected void onStop() {
        try {
            ortSession.endProfiling();
        } catch (OrtException e) {
            Log.e("ONNX", "Error ending profiling: " + e.getMessage());
            e.printStackTrace();
        }
        super.onStop();
    }

    @Override
    protected void onDestroy() {
        // Cleanup Text-to-Speech
        if (textToSpeech != null) {
            textToSpeech.stop();
            textToSpeech.shutdown();
        }

        // Cleanup ONNX resources
        try {
            ortSession.close();
            ortEnvironment.close();
        } catch (OrtException e) {
            Log.e("ONNX", "Error closing ONNX resources: " + e.getMessage());
            e.printStackTrace();
        }

        super.onDestroy();
    }
}