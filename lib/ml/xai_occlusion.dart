import 'dart:math';
import 'dart:typed_data';
import 'package:image/image.dart' as img;

import 'image_preprocess.dart';
import 'tomato_classifier.dart';

class OcclusionXAI {
  static const int imgSize = 224;

  static Future<Float32List> explain({
    required Uint8List originalImageBytes,
    required TomatoClassifier clf,
    required int targetClass,
    int grid = 8,
  }) async {
    final baseImg = ImagePreprocess.decodeCropResize(originalImageBytes);

    final baseline = clf.runProbsForImage(baseImg)[targetClass];

    final patch = (imgSize / grid).floor();
    final scores = List<double>.filled(grid * grid, 0.0);

    int idx = 0;
    for (int gy = 0; gy < grid; gy++) {
      for (int gx = 0; gx < grid; gx++) {
        final occluded = img.Image.from(baseImg);

        final xStart = gx * patch;
        final yStart = gy * patch;
        final xEnd = min(xStart + patch, imgSize);
        final yEnd = min(yStart + patch, imgSize);

        for (int y = yStart; y < yEnd; y++) {
          for (int x = xStart; x < xEnd; x++) {
            occluded.setPixelRgba(x, y, 128, 128, 128, 255);
          }
        }

        final score = clf.runProbsForImage(occluded)[targetClass];
        scores[idx++] = (baseline - score);
      }
    }

    double minV = scores.reduce(min);
    double maxV = scores.reduce(max);
    final denom = (maxV - minV).abs() < 1e-9 ? 1.0 : (maxV - minV);

    final heatmap = Float32List(imgSize * imgSize);
    for (int gy = 0; gy < grid; gy++) {
      for (int gx = 0; gx < grid; gx++) {
        final v = ((scores[gy * grid + gx] - minV) / denom).clamp(0.0, 1.0);
        final xStart = gx * patch;
        final yStart = gy * patch;
        final xEnd = min(xStart + patch, imgSize);
        final yEnd = min(yStart + patch, imgSize);

        for (int y = yStart; y < yEnd; y++) {
          for (int x = xStart; x < xEnd; x++) {
            heatmap[y * imgSize + x] = v.toDouble();
          }
        }
      }
    }

    return heatmap;
  }
}
