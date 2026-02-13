import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart'; // Clipboard
import 'package:image_picker/image_picker.dart';

import 'package:media_store_plus/media_store_plus.dart'; // Export to Downloads
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';

import '../ml/tomato_classifier.dart';
import '../ml/gate_classifier.dart';
import '../ml/xai_occlusion.dart';
import '../ml/heatmap_overlay.dart';
import '../ml/model_variants.dart';
import '../ml/run_logger.dart';

class MvpScreen extends StatefulWidget {
  const MvpScreen({super.key});

  @override
  State<MvpScreen> createState() => _MvpScreenState();
}

class _MvpScreenState extends State<MvpScreen> {
  final _picker = ImagePicker();

  // Models
  final _gate = GateClassifier();
  final _clf = TomatoClassifier();

  // Image + outputs
  Uint8List? _imgBytes;
  Uint8List? _overlayBytes;

  GateResult? _gateRes;
  Prediction? _pred;

  // UI states
  bool _loading = true; // initial load
  bool _busy = false;
  String _busyText = "";

  // timings
  double? _gateMs;
  double? _inferMs;
  double? _xaiMs;

  // controls
  ModelVariant _variant = ModelVariant.dynamic;

  // Gate parameters (tune if needed)
  double _gateTau = 0.55;
  int _gateTomatoIndex = 1; // if gate outputs [nonTomato, tomato]

  // XAI controls
  double _alpha = 0.45;
  int _grid = 8;

  int _jobId = 0; // cancels old async jobs

  @override
  void initState() {
    super.initState();
    _init();
  }

  void _setBusy(bool v, {String text = ""}) {
    if (!mounted) return;
    setState(() {
      _busy = v;
      _busyText = text;
    });
  }

  Future<void> _init() async {
    setState(() => _loading = true);
    try {
      // MediaStore init (needed once)
      await MediaStore.ensureInitialized();

      await _gate.load();
      await _clf.load(variant: _variant);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Init failed: $e")),
      );
    } finally {
      if (!mounted) return;
      setState(() => _loading = false);
    }
  }

  Future<void> _changeModel(ModelVariant v) async {
    _jobId++;
    _setBusy(true, text: "Loading disease model...");

    setState(() {
      _variant = v;
      _pred = null;
      _overlayBytes = null;
      _inferMs = null;
      _xaiMs = null;
    });

    try {
      await _clf.load(variant: v);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Model load failed: $e")),
      );
    } finally {
      _setBusy(false);
    }
  }

  Future<void> _pick(ImageSource src) async {
    _jobId++;
    final x = await _picker.pickImage(source: src, imageQuality: 95);
    if (x == null) return;

    final bytes = await x.readAsBytes();
    setState(() {
      _imgBytes = bytes;
      _overlayBytes = null;

      _gateRes = null;
      _pred = null;

      _gateMs = null;
      _inferMs = null;
      _xaiMs = null;
    });
  }

  Future<void> _predict() async {
    if (_imgBytes == null) return;

    _jobId++;
    final myJob = _jobId;

    _setBusy(true, text: "Checking tomato leaf (Gate)...");

    try {
      // --- 1) Gate ---
      final swGate = Stopwatch()..start();
      final g = _gate.predict(
        _imgBytes!,
        tau: _gateTau,
        tomatoIndex: _gateTomatoIndex,
      );
      swGate.stop();
      final gateMs = swGate.elapsedMicroseconds / 1000.0;

      if (!mounted || myJob != _jobId) return;
      setState(() {
        _gateRes = g;
        _gateMs = gateMs;

        // reset downstream outputs until disease runs
        _pred = null;
        _overlayBytes = null;
        _inferMs = null;
        _xaiMs = null;
      });

      if (!g.isTomatoLeaf) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              "🚫 Gate rejected: ${(100 * g.tomatoScore).toStringAsFixed(1)}% tomato. ${g.note}",
            ),
          ),
        );

        // log rejection
        await RunLogger.append({
          "ts": DateTime.now().toIso8601String(),
          "device": Platform.operatingSystem,
          "variant": _variant.title,
          "tau": _clf.policy.tau,
          "gate_tau": _gateTau,
          "margin": _clf.policy.margin,
          "grid": _grid,
          "alpha": _alpha,
          "pred_label": "GATE_REJECT",
          "pred_conf": g.tomatoScore,
          "abstained": true,
          "infer_ms": "",
          "xai_ms": "",
          "image_note": "gate_tau=$_gateTau gate_probs=${g.probs}",
        });

        return;
      }

      // --- 2) Disease ---
      _setBusy(true, text: "Predicting disease...");

      final sw = Stopwatch()..start();
      final p = _clf.predict(_imgBytes!);
      sw.stop();
      final inferMs = sw.elapsedMicroseconds / 1000.0;

      if (!mounted || myJob != _jobId) return;
      setState(() {
        _pred = p;
        _inferMs = inferMs;
        _overlayBytes = null;
        _xaiMs = null;
      });

      await RunLogger.append({
        "ts": DateTime.now().toIso8601String(),
        "device": Platform.operatingSystem,
        "variant": _variant.title,
        "tau": _clf.policy.tau,
        "gate_tau": _gateTau,
        "margin": _clf.policy.margin,
        "grid": _grid,
        "alpha": _alpha,
        "pred_label": p.label,
        "pred_conf": p.confidence,
        "abstained": p.abstained,
        "infer_ms": inferMs,
        "xai_ms": "",
        "image_note": "gate_ok score=${g.tomatoScore} gate_ms=$gateMs",
      });
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Predict error: $e")),
      );
    } finally {
      _setBusy(false);
    }
  }

  Future<void> _explain() async {
    if (_imgBytes == null || _pred == null) return;

    // Gate must be OK
    if (_gateRes == null || !_gateRes!.isTomatoLeaf) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Gate rejected image. XAI only runs for tomato leaves.")),
      );
      return;
    }

    // If abstained, don't run XAI
    if (_pred!.abstained) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Uncertain prediction. Retake photo for XAI.")),
      );
      return;
    }

    final myJob = ++_jobId;
    _setBusy(true, text: "Explaining (XAI)...");

    try {
      final sw = Stopwatch()..start();

      final heatmap = await OcclusionXAI.explain(
        originalImageBytes: _imgBytes!,
        clf: _clf,
        targetClass: _pred!.index,
        grid: _grid,
      );

      sw.stop();
      final xaiMs = sw.elapsedMicroseconds / 1000.0;

      if (!mounted || myJob != _jobId) return;

      final overlay = overlayHeatmapV2(
        originalBytes: _imgBytes!,
        heatmap: heatmap,
        alpha: _alpha,
        clipLow: 0.55,
      );

      if (!mounted || myJob != _jobId) return;
      setState(() {
        _overlayBytes = overlay;
        _xaiMs = xaiMs;
      });

      await RunLogger.append({
        "ts": DateTime.now().toIso8601String(),
        "device": Platform.operatingSystem,
        "variant": _variant.title,
        "tau": _clf.policy.tau,
        "gate_tau": _gateTau,
        "margin": _clf.policy.margin,
        "grid": _grid,
        "alpha": _alpha,
        "pred_label": _pred!.label,
        "pred_conf": _pred!.confidence,
        "abstained": _pred!.abstained,
        "infer_ms": _inferMs ?? "",
        "xai_ms": xaiMs,
        "image_note": "xai_done gate_score=${_gateRes!.tomatoScore}",
      });

      final path = await RunLogger.path();
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("✅ Logged runs.csv: $path")),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("XAI error: $e")),
      );
    } finally {
      _setBusy(false);
    }
  }

  // ---------------- CSV helpers (Option 1) ----------------

  Future<String> _csvPath() async => await RunLogger.path();

  Future<void> _showCsvPathDialog() async {
    final path = await _csvPath();
    if (!mounted) return;
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text("runs.csv location (App storage)"),
        content: SelectableText(path),
        actions: [
          TextButton(
            onPressed: () async {
              await Clipboard.setData(ClipboardData(text: path));
              if (!mounted) return;
              Navigator.pop(context);
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text("✅ Path copied")),
              );
            },
            child: const Text("Copy"),
          ),
          TextButton(onPressed: () => Navigator.pop(context), child: const Text("Close")),
        ],
      ),
    );
  }

  Future<void> _copyCsvPath() async {
    final path = await _csvPath();
    await Clipboard.setData(ClipboardData(text: path));
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text("✅ CSV path copied")),
    );
  }

  Future<void> _exportCsvToDownloads() async {
    _setBusy(true, text: "Exporting CSV to Downloads...");

    try {
      final srcPath = await _csvPath();
      final srcFile = File(srcPath);

      if (!await srcFile.exists()) {
        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("runs.csv not found yet. Make at least 1 prediction first.")),
        );
        return;
      }

      // Create a temp copy (MediaStore.saveFile will delete the temp file after saving)
      final cacheDir = await getTemporaryDirectory();
      final tempPath = p.join(
        cacheDir.path,
        "runs_${DateTime.now().millisecondsSinceEpoch}.csv",
      );
      await srcFile.copy(tempPath);

      final mediaStore = MediaStore();
      MediaStore.appFolder = "TomatoXAI"; // folder name inside Downloads

      final saved = await mediaStore.saveFile(
        tempFilePath: tempPath,
        dirType: DirType.download,
        dirName: DirName.download,
        relativePath: "TomatoXAI",
      );

      if (!mounted) return;

      if (saved == null) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("❌ Export failed (permission or storage issue).")),
        );
        return;
      }

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("✅ Exported to Downloads/TomatoXAI (file: ${saved.name})")),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Export error: $e")),
      );
    } finally {
      _setBusy(false);
    }
  }

  Future<void> _shareCsv() async {
    final srcPath = await _csvPath();
    final f = File(srcPath);

    if (!await f.exists()) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("runs.csv not found yet. Make at least 1 prediction first.")),
      );
      return;
    }

    await Share.shareXFiles(
      [XFile(srcPath)],
      text: "Tomato XAI runs.csv (gate + disease + XAI timings)",
    );
  }

  // --------------------------------------------------------

  @override
  void dispose() {
    _gate.close();
    _clf.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Stack(
      children: [
        Scaffold(
          appBar: AppBar(
            title: const Text("Tomato XAI MVP (Gate + Disease)"),
            actions: [
              PopupMenuButton<String>(
                onSelected: (v) async {
                  if (v == "show") await _showCsvPathDialog();
                  if (v == "copy") await _copyCsvPath();
                  if (v == "export") await _exportCsvToDownloads();
                  if (v == "share") await _shareCsv();
                },
                itemBuilder: (_) => const [
                  PopupMenuItem(value: "show", child: Text("Show CSV path")),
                  PopupMenuItem(value: "copy", child: Text("Copy CSV path")),
                  PopupMenuItem(value: "export", child: Text("Export CSV to Downloads")),
                  PopupMenuItem(value: "share", child: Text("Share runs.csv")),
                ],
              ),
            ],
          ),
          body: Padding(
            padding: const EdgeInsets.all(12),
            child: Column(
              children: [
                // Model selector
                Row(
                  children: [
                    const Text("Disease model: "),
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
                  const SizedBox(height: 4),
                  // ✅ Show loaded temperature
                  if (_clf.interpreter != null)
                    Text(
                      "Temperature scaling: T=${_clf.temperature.toStringAsFixed(2)}",
                      style: const TextStyle(fontSize: 12, color: Colors.grey),
                  ),

                const SizedBox(height: 8),

                // Gate threshold slider
                Row(
                  children: [
                    const Text("Gate τ"),
                    Expanded(
                      child: Slider(
                        value: _gateTau,
                        min: 0.50,
                        max: 0.95,
                        divisions: 45,
                        label: _gateTau.toStringAsFixed(2),
                        onChanged: _busy ? null : (v) => setState(() => _gateTau = v),
                      ),
                    ),
                    SizedBox(width: 54, child: Text(_gateTau.toStringAsFixed(2))),
                  ],
                ),

                const SizedBox(height: 8),

                // Image pick buttons
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

                // Image view
                Expanded(
                  child: Center(
                    child: (_imgBytes == null)
                        ? const Text("Pick an image to start")
                        : Image.memory(_overlayBytes ?? _imgBytes!, fit: BoxFit.contain),
                  ),
                ),

                // Gate status
                if (_gateRes != null) ...[
                  Text(
                    "Gate: ${_gateRes!.isTomatoLeaf ? "Tomato leaf ✅" : "Not tomato ❌"}"
                        " • ${(100 * _gateRes!.tomatoScore).toStringAsFixed(1)}%"
                        "${_gateMs != null ? " • ${_gateMs!.toStringAsFixed(1)} ms" : ""}",
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 6),
                ],

                // Prediction status
                if (_pred != null) ...[
                  Text(
                    "${_pred!.label}  •  ${(100 * _pred!.confidence).toStringAsFixed(1)}%",
                    style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 4),
                  if (_inferMs != null) Text("Inference: ${_inferMs!.toStringAsFixed(1)} ms"),
                  if (_xaiMs != null) Text("XAI: ${_xaiMs!.toStringAsFixed(1)} ms"),
                  if (_pred!.abstained && _pred!.note != null)
                    Padding(
                      padding: const EdgeInsets.only(top: 6),
                      child: Text(
                        _pred!.note!,
                        style: const TextStyle(color: Colors.redAccent),
                        textAlign: TextAlign.center,
                      ),
                    ),
                  const SizedBox(height: 6),
                ],

                // Predict + Explain
                Row(
                  children: [
                    Expanded(
                      child: ElevatedButton(
                        onPressed: (_imgBytes == null || _busy) ? null : _predict,
                        child: const Text("Predict"),
                      ),
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: ElevatedButton(
                        onPressed: (_pred == null || _busy) ? null : _explain,
                        child: const Text("Explain (XAI)"),
                      ),
                    ),
                  ],
                ),

                const SizedBox(height: 8),

                // XAI grid slider
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
                        onChanged: _busy ? null : (v) => setState(() => _grid = v.round()),
                      ),
                    ),
                  ],
                ),

                // Overlay alpha slider (only if overlay exists)
                if (_overlayBytes != null) ...[
                  Row(
                    children: [
                      const Text("Overlay α"),
                      Expanded(
                        child: Slider(
                          value: _alpha,
                          min: 0.1,
                          max: 0.8,
                          onChanged: _busy ? null : (v) => setState(() => _alpha = v),
                        ),
                      ),
                    ],
                  ),
                ],
              ],
            ),
          ),
        ),

        // ✅ Loading overlay
        if (_busy)
          Positioned.fill(
            child: Container(
              color: Colors.black.withOpacity(0.35),
              child: Center(
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 16),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(14),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const SizedBox(
                        width: 22,
                        height: 22,
                        child: CircularProgressIndicator(strokeWidth: 3),
                      ),
                      const SizedBox(width: 12),
                      Text(
                        _busyText.isEmpty ? "Working..." : _busyText,
                        style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w600),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
      ],
    );
  }
}
