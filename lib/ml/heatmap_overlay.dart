import 'dart:typed_data';
import 'package:image/image.dart' as img;

Uint8List overlayHeatmap({
  required Uint8List originalBytes,
  required Float32List heatmap, // 224*224
  double alpha = 0.45,
}) {
  img.Image? image = img.decodeImage(originalBytes);
  if (image == null) throw Exception("Decode failed");
  image = img.bakeOrientation(image);

  // same crop/resize so overlay aligns
  final w = image.width;
  final h = image.height;
  final size = w < h ? w : h;
  final x0 = (w - size) ~/ 2;
  final y0 = (h - size) ~/ 2;

  final cropped = img.copyCrop(image, x: x0, y: y0, width: size, height: size);
  final base = img.copyResize(cropped, width: 224, height: 224);

  for (int y = 0; y < 224; y++) {
    for (int x = 0; x < 224; x++) {
      final v = heatmap[y * 224 + x]; // 0..1

      // simple red/yellow heat
      final r = (255 * v).toInt().clamp(0, 255);
      final g = (80 * (1 - v)).toInt().clamp(0, 255);
      final b = 0;

      final p = base.getPixel(x, y);

      final nr = (p.r * (1 - alpha) + r * alpha).toInt();
      final ng = (p.g * (1 - alpha) + g * alpha).toInt();
      final nb = (p.b * (1 - alpha) + b * alpha).toInt();

      base.setPixelRgba(x, y, nr, ng, nb, 255);
    }
  }

  return Uint8List.fromList(img.encodeJpg(base, quality: 90));
}
