package com.gplio.numbersurface;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.PermissionChecker;
import android.util.Log;
import android.view.SurfaceView;
import android.widget.Button;
import android.widget.TextView;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.view.Menu;
import android.view.MenuItem;

import static java.security.AccessController.getContext;


// todo: handle task cancellations?
// todo: do continuous inference on the drawing?
public class MainActivity extends AppCompatActivity {
    public static final String TAG = "MainActivity";
    private static final int INPUT_W = 28;
    private DrawingView drawingView;
    private TextView resultsView;

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        String path = Environment.getExternalStorageDirectory().getPath();
        loadEnv(path);

        drawingView = (DrawingView) findViewById(R.id.dv_drawing);
        resultsView = (TextView) findViewById(R.id.tv_result);

        final String resultsText = "Draw and tap Analyse";
        resultsView.setText(resultsText);

        Button clearButton = (Button) findViewById(R.id.btn_clear);
        clearButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                log("Clearing canvas");
                drawingView.clearCanvas();
                resultsView.setText(resultsText);
            }
        });

        Button processButton = (Button) findViewById(R.id.btn_process);
        processButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                log("Calling task");
                new InferenceAsyncTask().execute();
            }
        });


        Context applicationContext = getApplicationContext();
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (ActivityCompat.checkSelfPermission(applicationContext, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                String permissions[] = {Manifest.permission.READ_EXTERNAL_STORAGE};
                int REQUEST_CODE = 2222;
                requestPermissions(permissions, REQUEST_CODE);
            }
        }
    }

    private float[] getDrawing() {
        Bitmap bitmap = Bitmap.createScaledBitmap(drawingView.getBitmap(), INPUT_W, INPUT_W, false);

        int w = bitmap.getWidth();
        int h = bitmap.getHeight();

        StringBuilder sb = new StringBuilder();

        float[] pixels = new float[w * h];
        int idx = 0;
        for (int j = 0; j < h; j++) {
            for (int i = 0; i < w; i++) {
                int pixel = bitmap.getPixel(i,j);
                pixels[idx] = (Color.red(pixel) + Color.green(pixel) + Color.blue(pixel)) / 3.0f;;
                pixels[idx] = (255.0f - pixels[idx]) / 255.0f;

                if (pixels[idx] > 0.1) {
                    sb.append("#");
                } else {
                    sb.append("-");
                }

                ++idx;
            }
            sb.append("\n");
        }

        log(sb.toString());
        return pixels;
    }


    @Override
    protected void onStop() {
        super.onStop();
        log("Freeing environment");
        freeEnv();
    }

    private void log(String s) {
        Log.d(TAG, s);
    }

    public native int loadEnv(String folder);
    public native int infer(float[] data, int w, int h);
    public native void freeEnv();

    private class InferenceAsyncTask extends AsyncTask<Void, Integer, Integer> {
        public static final String TAG = "InferenceAsyncTask";

        private final String[] names = {
                "Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"
        };

        @Override
        protected Integer doInBackground(Void... params) {
            float[] drawing = getDrawing();
            return infer(drawing, INPUT_W, INPUT_W);
        }

        @Override
        protected void onPostExecute(Integer result) {
            if (result >= 0 && result < names.length) {
                resultsView.setText("Is it a: " + names[result]);
            } else {
                resultsView.setText("Is it a: " + result);
            }
        }

        private void log(String s) {
            Log.d(TAG, s);
        }
    }
}
