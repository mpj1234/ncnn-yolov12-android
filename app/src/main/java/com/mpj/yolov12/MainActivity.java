package com.mpj.yolov12;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.PixelFormat;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;

import com.mpj.yolov12.databinding.ActivityMainBinding;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

public class MainActivity extends AppCompatActivity implements SurfaceHolder.Callback {
	private static final String TAG = "MainActivity";

	public static final int REQUEST_CAMERA = 100;

	private final Yolov12 yolov12 = new Yolov12();

	private int facing = 0;
	private int current_model = 0;
	private int current_cpugpu = 0;
	private SurfaceView cameraView = null;
	/**
	 * true 启动检测
	 * false 关闭检测
	 */
	boolean stopStart = false;

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);

		getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

		cameraView = findViewById(R.id.cameraview);
		cameraView.getHolder().setFormat(PixelFormat.RGBA_8888);
		cameraView.getHolder().addCallback(this);

		Button buttonSwitchCamera = findViewById(R.id.buttonSwitchCamera);
		buttonSwitchCamera.setOnClickListener(arg0 -> {
			int new_facing = 1 - facing;

			yolov12.closeCamera();

			yolov12.openCamera(new_facing);
			stopStart = true;

			facing = new_facing;

			// cameraView设置为白色背景
			cameraView.setBackgroundColor(0x00000000);
		});

		Button buttonStopStart = findViewById(R.id.buttonStopStart);
		buttonStopStart.setOnClickListener(arg0 -> {
			if (!stopStart) {
				stopStart = true;

				yolov12.openCamera(facing);
			} else {
				stopStart = false;
				yolov12.closeCamera();
			}

			// cameraView设置为白色背景
			cameraView.setBackgroundColor(0x00000000);
		});

		Spinner spinnerModel = findViewById(R.id.spinnerModel);
		spinnerModel.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
			@Override
			public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id) {
				if (position != current_model) {
					current_model = position;
					reload();
				}
			}

			@Override
			public void onNothingSelected(AdapterView<?> arg0) {
			}
		});

		Spinner spinnerCPUGPU = findViewById(R.id.spinnerCPUGPU);
		spinnerCPUGPU.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
			@Override
			public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id) {
				if (position != current_cpugpu) {
					current_cpugpu = position;
					reload();
				}
			}

			@Override
			public void onNothingSelected(AdapterView<?> arg0) {
			}
		});

		// cameraView设置为白色背景
		cameraView.setBackgroundColor(0xFFFFFFFF);

		reload();
	}

	private void reload() {
		boolean ret_init = yolov12.loadModel(getAssets(), current_model, current_cpugpu);
		if (!ret_init) {
			Log.e(TAG, "loadModel failed");
		}
	}

	@Override
	public void surfaceCreated(@NonNull SurfaceHolder surfaceHolder) {
	}

	@Override
	public void surfaceChanged(@NonNull SurfaceHolder surfaceHolder, int i, int i1, int i2) {
		yolov12.setOutputWindow(surfaceHolder.getSurface());
	}

	@Override
	public void surfaceDestroyed(@NonNull SurfaceHolder surfaceHolder) {
	}

	@Override
	public void onResume() {
		super.onResume();

		if (ContextCompat.checkSelfPermission(getApplicationContext(), android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED) {
			ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA);
		}
	}

	@Override
	public void onPause() {
		super.onPause();

		yolov12.closeCamera();
		stopStart = false;
		// cameraView设置为白色背景
		if (cameraView != null) {
			cameraView.setBackgroundColor(0xFFFFFFFF);
		}
	}
}