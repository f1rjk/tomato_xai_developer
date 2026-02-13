import 'dart:typed_data';

extension ReshapeFloat32 on Float32List {
  List reshape(List<int> dims) {
    final b = dims[0], h = dims[1], w = dims[2], c = dims[3];
    int idx = 0;

    final out = List.generate(
      b,
          (_) => List.generate(
        h,
            (_) => List.generate(
          w,
              (_) => List<double>.filled(c, 0.0),
        ),
      ),
    );

    for (int bi = 0; bi < b; bi++) {
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          for (int ch = 0; ch < c; ch++) {
            out[bi][y][x][ch] = this[idx++];
          }
        }
      }
    }
    return out;
  }
}

extension ReshapeListDouble on List<double> {
  List reshape(List<int> dims) {
    final b = dims[0], n = dims[1];
    final out = List.generate(b, (_) => List<double>.filled(n, 0.0));
    int idx = 0;
    for (int bi = 0; bi < b; bi++) {
      for (int i = 0; i < n; i++) {
        out[bi][i] = this[idx++];
      }
    }
    return out;
  }
}
