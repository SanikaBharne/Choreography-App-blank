import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

import '../models/analysis_result.dart';
import '../models/media_asset.dart';
import '../models/pose_result.dart';
import '../services/api_client.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}


class _HomePageState extends State<HomePage> {
  final _baseUrlController = TextEditingController(text: 'http://127.0.0.1:8000');
  final _poseTimestampsController = TextEditingController();
  String _beatMethod = 'librosa';
  double _speedMultiplier = 0.75;
  int _subdivisions = 2;
  int _poseSamples = 4;
  bool _mirror = false;
  bool _loading = false;
  String? _status;
  MediaAsset? _asset;
  AnalysisSummary? _analysis;
  PoseResult? _pose;
  VideoPlayerController? _practiceController;


  @override
  void dispose() {
    _baseUrlController.dispose();
    _poseTimestampsController.dispose();
    _practiceController?.dispose();
    super.dispose();
  }

  ApiClient get _apiClient => ApiClient(baseUrl: _baseUrlController.text.trim());

  Future<void> _pickAndUpload() async {
    final result = await FilePicker.platform.pickFiles(withData: true, type: FileType.media);
    if (result == null || result.files.single.bytes == null) {
      return;
    }

    setState(() {
      _loading = true;
      _status = 'Uploading media...';
      _analysis = null;
      _pose = null;
    });

    try {
      final file = result.files.single;
      final asset = await _apiClient.uploadMedia(
        filename: file.name,
        bytes: file.bytes!,
      );
      setState(() {
        _asset = asset;
        _status = 'Uploaded ${asset.filename}';
      });
    } catch (error) {
      setState(() => _status = error.toString());
    } finally {
      setState(() => _loading = false);
    }
  }

  Future<void> _runAnalysis() async {
    final asset = _asset;
    if (asset == null) return;
    setState(() {
      _loading = true;
      _status = 'Analyzing beats...';
    });
    try {
      final analysis = await _apiClient.analyzeMedia(
        mediaId: asset.mediaId,
        beatMethod: _beatMethod,
        subdivisions: _subdivisions,
      );
      setState(() {
        _analysis = analysis;
        _status = 'Analysis complete';
      });
    } catch (error) {
      setState(() => _status = error.toString());
    } finally {
      setState(() => _loading = false);
    }
  }

  Future<void> _generatePracticeClip() async {
    final asset = _asset;
    if (asset == null || asset.mediaType != 'video') return;
    setState(() {
      _loading = true;
      _status = 'Generating practice clip...';
    });
    try {
      final previewUrl = await _apiClient.generatePracticeClip(
        mediaId: asset.mediaId,
        startSec: 0,
        endSec: 6,
        speedMultiplier: _speedMultiplier,
        mirror: _mirror,
      );
      await _practiceController?.dispose();
      final controller = VideoPlayerController.networkUrl(Uri.parse(previewUrl));
      await controller.initialize();
      setState(() {
        _practiceController = controller;
        _status = 'Practice clip ready';
      });
    } catch (error) {
      setState(() => _status = error.toString());
    } finally {
      setState(() => _loading = false);
    }
  }

  Future<void> _runPose() async {
    final asset = _asset;
    if (asset == null || asset.mediaType != 'video') return;
    setState(() {
      _loading = true;
      _status = 'Running pose estimation...';
    });
    try {
      List<double>? timestamps;
      final text = _poseTimestampsController.text.trim();
      if (text.isNotEmpty) {
        timestamps = text.split(',').map((s) => double.tryParse(s.trim())).whereType<double>().toList();
        if (timestamps.isEmpty) timestamps = null;
      }
      final pose = await _apiClient.analyzePose(
        mediaId: asset.mediaId,
        sampleCount: _poseSamples,
        timestamps: timestamps,
      );
      setState(() {
        _pose = pose;
        _status = 'Pose estimation complete';
      });
    } catch (error) {
      setState(() => _status = error.toString());
    } finally {
      setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _HeroCard(status: _status),
              const SizedBox(height: 16),
              TextField(
                controller: _baseUrlController,
                decoration: const InputDecoration(labelText: 'FastAPI base URL'),
              ),
              const SizedBox(height: 16),
              Wrap(
                spacing: 12,
                runSpacing: 12,
                children: [
                  FilledButton(
                    onPressed: _loading ? null : _pickAndUpload,
                    child: const Text('Upload Media'),
                  ),
                  FilledButton(
                    onPressed: _loading || _asset == null ? null : _runAnalysis,
                    child: const Text('Run Analysis'),
                  ),
                  FilledButton(
                    onPressed: _loading || _asset?.mediaType != 'video' ? null : _generatePracticeClip,
                    child: const Text('Practice Clip'),
                  ),
                  FilledButton(
                    onPressed: _loading || _asset?.mediaType != 'video' ? null : _runPose,
                    child: const Text('Pose Estimation'),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              TextField(
                controller: _poseTimestampsController,
                decoration: const InputDecoration(
                  labelText: 'Pose timestamps (comma-separated, seconds)',
                  hintText: 'e.g. 1.2, 3.5, 7.0',
                ),
                keyboardType: TextInputType.text,
              ),
              const SizedBox(height: 16),
              _ControlsRow(
                beatMethod: _beatMethod,
                subdivisions: _subdivisions,
                speedMultiplier: _speedMultiplier,
                mirror: _mirror,
                poseSamples: _poseSamples,
                onBeatMethodChanged: (value) => setState(() => _beatMethod = value),
                onSubdivisionsChanged: (value) => setState(() => _subdivisions = value),
                onSpeedChanged: (value) => setState(() => _speedMultiplier = value),
                onMirrorChanged: (value) => setState(() => _mirror = value),
                onPoseSamplesChanged: (value) => setState(() => _poseSamples = value),
              ),
              const SizedBox(height: 20),
              if (_asset != null) _SoftCard(child: Text('Current upload: ${_asset!.filename} (${_asset!.mediaType})')),
              if (_analysis != null) ...[
                const SizedBox(height: 20),
                _AnalysisSection(analysis: _analysis!),
              ],
              if (_practiceController != null) ...[
                const SizedBox(height: 20),
                _SoftCard(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text('Practice Clip', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700)),
                      const SizedBox(height: 12),
                      AspectRatio(
                        aspectRatio: _practiceController!.value.aspectRatio,
                        child: VideoPlayer(_practiceController!),
                      ),
                      const SizedBox(height: 12),
                      FilledButton(
                        onPressed: () => _practiceController!.value.isPlaying
                            ? _practiceController!.pause()
                            : _practiceController!.play(),
                        child: Text(_practiceController!.value.isPlaying ? 'Pause' : 'Play'),
                      ),
                    ],
                  ),
                ),
              ],
              if (_pose != null) ...[
                const SizedBox(height: 20),
                _PoseSection(baseUrl: _baseUrlController.text.trim(), pose: _pose!),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

class _HeroCard extends StatelessWidget {
  const _HeroCard({this.status});

  final String? status;

  @override
  Widget build(BuildContext context) {
    return _SoftCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Practice Studio', style: TextStyle(letterSpacing: 1.3, color: Color(0xFFA57C6D))),
          const SizedBox(height: 8),
          const Text(
            'Choreo App',
            style: TextStyle(fontSize: 32, fontWeight: FontWeight.w800, color: Color(0xFF62483F)),
          ),
          const SizedBox(height: 8),
          const Text(
            'Upload a song or dance video, run beat analysis, generate practice clips, and inspect pose snapshots.',
          ),
          if (status != null) ...[
            const SizedBox(height: 12),
            Text(status!, style: const TextStyle(color: Color(0xFF7D5F55))),
          ],
        ],
      ),
    );
  }
}

class _ControlsRow extends StatelessWidget {
  const _ControlsRow({
    required this.beatMethod,
    required this.subdivisions,
    required this.speedMultiplier,
    required this.mirror,
    required this.poseSamples,
    required this.onBeatMethodChanged,
    required this.onSubdivisionsChanged,
    required this.onSpeedChanged,
    required this.onMirrorChanged,
    required this.onPoseSamplesChanged,
  });

  final String beatMethod;
  final int subdivisions;
  final double speedMultiplier;
  final bool mirror;
  final int poseSamples;
  final ValueChanged<String> onBeatMethodChanged;
  final ValueChanged<int> onSubdivisionsChanged;
  final ValueChanged<double> onSpeedChanged;
  final ValueChanged<bool> onMirrorChanged;
  final ValueChanged<int> onPoseSamplesChanged;

  @override
  Widget build(BuildContext context) {
    return _SoftCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Session Controls', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700)),
          const SizedBox(height: 12),
          DropdownButtonFormField<String>(
            initialValue: beatMethod,
            decoration: const InputDecoration(labelText: 'Beat engine'),
            items: const [
              DropdownMenuItem(value: 'librosa', child: Text('Librosa')),
              DropdownMenuItem(value: 'scratch', child: Text('Scratch')),
            ],
            onChanged: (value) {
              if (value != null) onBeatMethodChanged(value);
            },
          ),
          const SizedBox(height: 12),
          Text('Subdivisions: $subdivisions'),
          Slider(value: subdivisions.toDouble(), min: 1, max: 4, divisions: 3, onChanged: (value) => onSubdivisionsChanged(value.round())),
          Text('Practice speed: ${speedMultiplier.toStringAsFixed(2)}x'),
          Slider(value: speedMultiplier, min: 0.25, max: 2.0, divisions: 35, onChanged: onSpeedChanged),
          Text('Pose samples: $poseSamples'),
          Slider(value: poseSamples.toDouble(), min: 2, max: 8, divisions: 6, onChanged: (value) => onPoseSamplesChanged(value.round())),
          SwitchListTile(
            contentPadding: EdgeInsets.zero,
            title: const Text('Mirror practice clip'),
            value: mirror,
            onChanged: onMirrorChanged,
          ),
        ],
      ),
    );
  }
}

class _AnalysisSection extends StatelessWidget {
  const _AnalysisSection({required this.analysis});

  final AnalysisSummary analysis;

  @override
  Widget build(BuildContext context) {
    return _SoftCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Beat Analysis', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700)),
          const SizedBox(height: 12),
          Wrap(
            spacing: 12,
            runSpacing: 12,
            children: [
              _MetricChip(label: 'Duration', value: '${analysis.duration.toStringAsFixed(1)}s'),
              _MetricChip(label: 'Tempo', value: analysis.tempoEstimate?.toStringAsFixed(1) ?? 'N/A'),
              _MetricChip(label: 'Beats', value: analysis.beats.length.toString()),
              _MetricChip(label: 'Engine', value: analysis.beatMethod),
            ],
          ),
          const SizedBox(height: 12),
          const Text('Eight-count summaries', style: TextStyle(fontWeight: FontWeight.w700)),
          const SizedBox(height: 8),
          ...analysis.eightCountSummaries.map(
            (summary) => Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Text('${summary['label']}: ${summary['values']}'),
            ),
          ),
        ],
      ),
    );
  }
}

class _PoseSection extends StatelessWidget {
  const _PoseSection({
    required this.baseUrl,
    required this.pose,
  });

  final String baseUrl;
  final PoseResult pose;

  @override
  Widget build(BuildContext context) {
    return _SoftCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Pose Estimation', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700)),
          const SizedBox(height: 12),
          Wrap(
            spacing: 12,
            runSpacing: 12,
            children: [
              _MetricChip(label: 'Frames', value: '${pose.overview['sample_count']}'),
              _MetricChip(label: 'Detections', value: '${pose.overview['detected_frames']}'),
              _MetricChip(label: 'Visibility', value: '${pose.overview['average_visibility']}'),
            ],
          ),
          const SizedBox(height: 12),
          ...pose.frames.map(
            (frame) => Padding(
              padding: const EdgeInsets.only(bottom: 16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Timestamp ${frame['timestamp']}s', style: const TextStyle(fontWeight: FontWeight.w700)),
                  const SizedBox(height: 8),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(18),
                    child: Image.network('$baseUrl${frame['overlay_url']}'),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _MetricChip extends StatelessWidget {
  const _MetricChip({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.9),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: const Color(0xFFD7C1B7)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label, style: const TextStyle(fontSize: 12, color: Color(0xFF9A7366))),
          const SizedBox(height: 2),
          Text(value, style: const TextStyle(fontWeight: FontWeight.w700)),
        ],
      ),
    );
  }
}

class _SoftCard extends StatelessWidget {
  const _SoftCard({required this.child});

  final Widget child;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.84),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: const Color(0xFFD9C7BE)),
        boxShadow: const [
          BoxShadow(
            color: Color(0x14000000),
            blurRadius: 20,
            offset: Offset(0, 10),
          ),
        ],
      ),
      child: child,
    );
  }
}
