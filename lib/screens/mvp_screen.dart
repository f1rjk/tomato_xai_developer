import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../ml/tomato_classifier.dart';
import '../ml/xai_occlusion.dart';
import '../ml/heatmap_overlay.dart';
import '../ml/model_variants.dart';

class MvpScreen extends StatefulWidget {
  const MvpScreen({super.key});

  @override
  State<MvpScreen> createState() => _MvpScreenState();
}

class _MvpScreenState extends State<MvpScreen> {
  final _picker = ImagePicker();
  final _clf = TomatoClassifier();

  Uint8List? _imgBytes;
  Uint8List? _overlayBytes;
  Prediction? _pred;

  bool _loadingModel = true;
  bool _busy = false;

  double _alpha = 0.45;
  int _grid = 8;

  ModelVariant _variant = ModelVariant.dynamic;

  int _jobId = 0; // cancel old work

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    await _clf.load(variant: _variant);
    if (!mounted) return;
    setState(() => _loadingModel = false);
  }

  Future<void> _changeModel(ModelVariant v) async {
    setState(() {
      _busy = true;
      _variant = v;
      _pred = null;
      _overlayBytes = null;
    });

    try {
      await _clf.load(variant: v);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Model load failed: $e")),
      );
    } finally {
      if (!mounted) return;
      setState(() => _busy = false);
    }
  }

  Future<void> _pick(ImageSource src) async {
    final x = await _picker.pickImage(source: src, imageQuality: 95);
    if (x == null) return;

    final bytes = await x.readAsBytes();
    setState(() {
      _imgBytes = bytes;
      _overlayBytes = null;
      _pred = null;
    });
  }

  Future<void> _predict() async {
    if (_imgBytes == null) return;

    setState(() => _busy = true);

    try {
      final p = _clf.predict(_imgBytes!);
      if (!mounted) return;
      setState(() => _pred = p);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Predict error: $e")),
      );
    } finally {
      if (!mounted) return;
      setState(() => _busy = false);
    }
  }

  Future<void> _explain() async {
    if (_imgBytes == null || _pred == null) return;

    // If abstained, do not run XAI
    if (_pred!.abstained) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Uncertain prediction. Retake photo for XAI.")),
      );
      return;
    }

    final myJob = ++_jobId;

    setState(() => _busy = true);

    try {
      final heatmap = await OcclusionXAI.explain(
        originalImageBytes: _imgBytes!,
        clf: _clf,
        targetClass: _pred!.index,
        grid: _grid,
      );

      // if user started new job, ignore old result
      if (myJob != _jobId) return;

      final overlay = overlayHeatmapV2(
        originalBytes: _imgBytes!,
        heatmap: heatmap,
        alpha: _alpha,
        clipLow: 0.55,
      );

      if (!mounted) return;
      setState(() => _overlayBytes = overlay);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("XAI error: $e")),
      );
    } finally {
      if (!mounted) return;
      setState(() => _busy = false);
    }
  }

  @override
  void dispose() {
    _clf.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_loadingModel) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      appBar: AppBar(title: const Text("Tomato XAI MVP (v2)")),
      body: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          children: [
            // Model selector
            Row(
              children: [
                const Text("Model: "),
                const SizedBox(width: 10),
                Expanded(
                  child: DropdownButton<ModelVariant>(
                    value: _variant,
                    isExpanded: true,
                    onChanged: _busy ? null : (v) => _changeModel(v!),
                    items: ModelVariant.values
                        .map((v) => DropdownMenuItem(
                      value: v,
                      child: Text(v.title),
                    ))
                        .toList(),
                  ),
                ),
              ],
            ),

            const SizedBox(height: 10),

            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _busy ? null : () => _pick(ImageSource.gallery),
                    icon: const Icon(Icons.photo),
                    label: const Text("Gallery"),
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _busy ? null : () => _pick(ImageSource.camera),
                    icon: const Icon(Icons.camera_alt),
                    label: const Text("Camera"),
                  ),
                ),
              ],
            ),

            const SizedBox(height: 12),

            Expanded(
              child: Center(
                child: (_imgBytes == null)
                    ? const Text("Pick an image to start")
                    : Image.memory(_overlayBytes ?? _imgBytes!, fit: BoxFit.contain),
              ),
            ),

            if (_pred != null) ...[
              Text(
                "${_pred!.label}  •  ${(100 * _pred!.confidence).toStringAsFixed(1)}%",
                style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
              ),
              if (_pred!.abstained && _pred!.note != null)
                Padding(
                  padding: const EdgeInsets.only(top: 6),
                  child: Text(
                    _pred!.note!,
                    style: const TextStyle(color: Colors.redAccent),
                    textAlign: TextAlign.center,
                  ),
                ),
            ],

            const SizedBox(height: 10),

            Row(
              children: [
                Expanded(
                  child: ElevatedButton(
                    onPressed: (_imgBytes == null || _busy) ? null : _predict,
                    child: Text(_busy ? "Working..." : "Predict"),
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: ElevatedButton(
                    onPressed: (_pred == null || _busy) ? null : _explain,
                    child: Text(_busy ? "Working..." : "Explain (XAI)"),
                  ),
                ),
              ],
            ),

            const SizedBox(height: 8),

            Row(
              children: [
                const Text("XAI grid"),
                Expanded(
                  child: Slider(
                    value: _grid.toDouble(),
                    min: 5,
                    max: 10,
                    divisions: 5,
                    label: "grid=$_grid",
                    onChanged: (v) => setState(() => _grid = v.round()),
                  ),
                ),
              ],
            ),

            if (_overlayBytes != null) ...[
              Row(
                children: [
                  const Text("Overlay"),
                  Expanded(
                    child: Slider(
                      value: _alpha,
                      min: 0.1,
                      max: 0.8,
                      onChanged: (v) => setState(() => _alpha = v),
                    ),
                  ),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }
}
