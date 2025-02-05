package com.example.yolov8_detect;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;
import androidx.annotation.Nullable;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class RectView extends View {
    private final Map<RectF, String> detectionMap = new HashMap<>();
    private final Paint[] classPaints;
    private final Paint textPaint = new Paint();
    private String[] labels;

    public RectView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);

        // Inisialisasi paint untuk setiap class dengan warna berbeda
        classPaints = new Paint[9]; // Sesuai jumlah class
        for(int i = 0; i < classPaints.length; i++) {
            classPaints[i] = new Paint();
            classPaints[i].setStyle(Paint.Style.STROKE);
            classPaints[i].setStrokeWidth(10.0f);
            classPaints[i].setStrokeCap(Paint.Cap.ROUND);
            classPaints[i].setStrokeJoin(Paint.Join.ROUND);
            classPaints[i].setStrokeMiter(100);
        }

        // Set warna untuk setiap class dengan warna yang lebih kontras
        classPaints[0].setColor(Color.rgb(65, 105, 225));    // GedungFTMM - Royal Blue
        classPaints[1].setColor(Color.rgb(220, 20, 60));     // Mobil - Crimson Red
        classPaints[2].setColor(Color.rgb(50, 205, 50));     // RambuBelokKanan - Lime Green
        classPaints[3].setColor(Color.rgb(255, 215, 0));     // RambuBelokKiri - Golden Yellow
        classPaints[4].setColor(Color.rgb(0, 255, 255));     // RambuPutarBalik - Aqua
        classPaints[5].setColor(Color.rgb(255, 0, 255));     // RambuTitikKumpul - Magenta
        classPaints[6].setColor(Color.rgb(255, 255, 255));   // ZebraCross - White
        classPaints[7].setColor(Color.rgb(255, 165, 0));     // orang - Orange
        classPaints[8].setColor(Color.rgb(128, 0, 128));     // sepedamotor - Purple

        // Konfigurasi paint untuk teks
        textPaint.setTextSize(60.0f);
        textPaint.setColor(Color.WHITE);
        textPaint.setShadowLayer(5.0f, 0f, 0f, Color.BLACK); // Menambah shadow untuk keterbacaan
    }

    public void setLabels(String[] labels) {
        this.labels = labels;
    }

    // Method untuk transformasi koordinat rectangle
    public ArrayList<Result> transFormRect(ArrayList<Result> resultArrayList) {
        float scaleX = getWidth() / (float) SupportOnnx.INPUT_SIZE;
        float scaleY = scaleX * 9f / 16f;
        float realY = getWidth() * 9f / 16f;
        float diffY = realY - getHeight();

        for (Result result : resultArrayList) {
            RectF rect = result.getRectF();
            rect.left *= scaleX;
            rect.right *= scaleX;
            rect.top = rect.top * scaleY - (diffY / 2f);
            rect.bottom = rect.bottom * scaleY - (diffY / 2f);
        }
        return resultArrayList;
    }

    // Method untuk membersihkan deteksi sebelumnya
    public void clear() {
        detectionMap.clear();
    }

    // Method untuk mengubah hasil deteksi menjadi map untuk ditampilkan
    public void resultToList(ArrayList<Result> results) {
        for (Result result : results) {
            String label = labels[result.getLabel()];
            int confidence = Math.round(result.getScore() * 100);
            String displayText = String.format("%s, %d%%", label, confidence);
            detectionMap.put(result.getRectF(), displayText);
        }
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        for (Map.Entry<RectF, String> detection : detectionMap.entrySet()) {
            RectF rect = detection.getKey();
            String text = detection.getValue();
            String label = text.split(",")[0].trim();

            // Mendapatkan index class untuk warna yang sesuai
            int classIndex = getClassIndex(label);

            // Menggambar rectangle
            canvas.drawRect(rect, classPaints[classIndex]);

            // Menggambar background untuk teks
            Paint bgPaint = new Paint();
            bgPaint.setColor(Color.argb(160, 0, 0, 0)); // Semi-transparent black
            float textWidth = textPaint.measureText(text);
            float textHeight = textPaint.getTextSize();
            canvas.drawRect(
                    rect.left + 10.0f,
                    rect.top,
                    rect.left + textWidth + 20.0f,
                    rect.top + textHeight + 10.0f,
                    bgPaint
            );

            // Menggambar teks
            canvas.drawText(text,
                    rect.left + 10.0f,
                    rect.top + textHeight,
                    textPaint);
        }
    }

    private int getClassIndex(String label) {
        if (labels != null) {
            for (int i = 0; i < labels.length; i++) {
                if (labels[i].equals(label)) return i;
            }
        }
        return 0;
    }
}