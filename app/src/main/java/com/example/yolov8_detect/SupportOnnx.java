package com.example.yolov8_detect;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.YuvImage;
import android.media.Image;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;

public class SupportOnnx {
    static final String fileName = "cobaa.onnx";
    static final String labelName = "label.txt";
    static final int INPUT_SIZE = 640;
    static final int BATCH_SIZE = 1;
    static final int PIXEL_SIZE = 3;
    static final int FLOAT_SIZE = 4;

    public float iouThresh = 0.5f;
    public float objectThresh = 0.4f;
    private final Context context;
    private String[] labels;

    public SupportOnnx(Context context) {
        this.context = context;
    }

    public void loadModel() {
        AssetManager assetManager = context.getAssets();
        File outputFile = new File(context.getFilesDir() + "/" + fileName);

        try {
            InputStream inputStream = assetManager.open(fileName);
            OutputStream outputStream = new FileOutputStream(outputFile);
            byte[] buffer = new byte[1024 * 4];
            int read;
            while ((read = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, read);
            }

            inputStream.close();
            outputStream.flush();
            outputStream.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void loadLabel() {
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(context.getAssets().open(labelName)));
            String line;
            List<String> labelList = new ArrayList<>();
            while ((line = reader.readLine()) != null) {
                labelList.add(line);
            }
            labels = new String[labelList.size()];
            labelList.toArray(labels);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public Bitmap imageToBitmap(Image image) {
        Image.Plane[] planes = image.getPlanes();

        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();
        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();
        byte[] yuvBytes = new byte[ySize + uSize + vSize];
        yBuffer.get(yuvBytes, 0, ySize);
        vBuffer.get(yuvBytes, ySize, vSize);
        uBuffer.get(yuvBytes, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(yuvBytes, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 90, out);

        byte[] imageBytes = out.toByteArray();

        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    public Bitmap rescaleBitmap(Bitmap bitmap) {
        return Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);
    }

    public FloatBuffer bitmapToFloatBuffer(Bitmap bitmap) {
        int cap = BATCH_SIZE * PIXEL_SIZE * INPUT_SIZE * INPUT_SIZE;
        ByteOrder order = ByteOrder.nativeOrder();
        FloatBuffer buffer = ByteBuffer.allocate(cap * FLOAT_SIZE).order(order).asFloatBuffer();

        int area = INPUT_SIZE * INPUT_SIZE;
        int[] bitmapData = new int[area];

        bitmap.getPixels(bitmapData, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        for (int i = 0; i < INPUT_SIZE - 1; i++) {
            for (int j = 0; j < INPUT_SIZE - 1; j++) {
                int idx = INPUT_SIZE * i + j;
                int pixelValue = bitmapData[idx];
                float imageMean = 0.0f;
                float imageSTD = 255.0f;
                buffer.put(idx, (((pixelValue >> 16 & 0xFF) - imageMean) / imageSTD));
                buffer.put(idx + area, (((pixelValue >> 8 & 0xFF) - imageMean) / imageSTD));
                buffer.put(idx + area * 2, (((pixelValue & 0xFF) - imageMean) / imageSTD));
            }
        }
        buffer.rewind();
        return buffer;
    }

    public ArrayList<Result> outputsToNMSPredictions(float[][][] output, int rows) {
        ArrayList<Result> results = new ArrayList<>();

        float[][][] outputV8 = new float[1][rows][output[0].length];

        for (int l = 0; l < output[0].length ; l++) {
            for (int m = 0; m < rows; m++) {
                outputV8[0][m][l] = output[0][l][m];
            }
        }


        for (int i = 0; i < rows; ++i) {
            int detectionClass = -1;
            float maxClass = 0;

            float[] _classes = new float[labels.length];
            System.arraycopy(outputV8[0][i], 4, _classes, 0, labels.length);

            for (int c = 0; c < labels.length; ++c) {
                if (_classes[c] > maxClass) {
                    detectionClass = c;
                    maxClass = _classes[c];
                }
            }

            float confidenceInClass = maxClass;
            if (confidenceInClass > objectThresh) {
                float xPos = outputV8[0][i][0];
                float yPos = outputV8[0][i][1];
                float width = outputV8[0][i][2];
                float height = outputV8[0][i][3];

                RectF rectF = new RectF(Math.max(0, xPos - width / 2), Math.max(0, yPos - height / 2),
                        Math.min(INPUT_SIZE - 1, xPos + width / 2), Math.min(INPUT_SIZE - 1, yPos + height / 2));
                Result recognition = new Result(detectionClass, confidenceInClass, rectF);
                results.add(recognition);
            }
        }

        return nms(results);
    }

    public ArrayList<Result> nms(ArrayList<Result> results) {
        ArrayList<Result> nmsList = new ArrayList<>();

        for (int k = 0; k < labels.length; k++) {
            //1.find max confidence per class
            PriorityQueue<Result> pq =
                    new PriorityQueue<>(50,
                            (o1, o2) -> Float.compare(o1.getScore(), o2.getScore()));

            for (int i = 0; i < results.size(); i++) {
                if (results.get(i).getLabel() == k) {
                    pq.add(results.get(i));
                }
            }

            //2.do non maximum suppression
            while (pq.size() > 0) {
                //insert detection with max confidence
                Result[] a = new Result[pq.size()];
                Result[] detections = pq.toArray(a);
                Result max = detections[0];
                nmsList.add(max);
                pq.clear();

                for (int j = 1; j < detections.length; j++) {
                    Result detection = detections[j];
                    RectF b = detection.getRectF();
                    if (box_iou(max.getRectF(), b) < iouThresh) {
                        pq.add(detection);
                    }
                }
            }
        }
        return nmsList;
    }

    protected float box_iou(RectF a, RectF b) {
        return box_intersection(a, b) / box_union(a, b);
    }

    protected float box_intersection(RectF a, RectF b) {
        float w = overlap((a.left + a.right) / 2, a.right - a.left,
                (b.left + b.right) / 2, b.right - b.left);
        float h = overlap((a.top + a.bottom) / 2, a.bottom - a.top,
                (b.top + b.bottom) / 2, b.bottom - b.top);
        if (w < 0 || h < 0) return 0;
        return w * h;
    }

    protected float box_union(RectF a, RectF b) {
        float i = box_intersection(a, b);
        return (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i;
    }

    protected float overlap(float x1, float w1, float x2, float w2) {
        float l1 = x1 - w1 / 2;
        float l2 = x2 - w2 / 2;
        float left = Math.max(l1, l2);
        float r1 = x1 + w1 / 2;
        float r2 = x2 + w2 / 2;
        float right = Math.min(r1, r2);
        return right - left;
    }

    public String[] getLabels() {
        return labels;
    }
}
