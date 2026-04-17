import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

import '../models/analysis_result.dart';
import '../models/media_asset.dart';
import '../models/pose_result.dart';

class ApiClient {
  ApiClient({required this.baseUrl});

  final String baseUrl;

  Uri _uri(String path) => Uri.parse('$baseUrl$path');

  Future<MediaAsset> uploadMedia({
    required String filename,
    required Uint8List bytes,
  }) async {
    final request = http.MultipartRequest('POST', _uri('/api/media/upload'))
      ..files.add(
        http.MultipartFile.fromBytes('file', bytes, filename: filename),
      );
    final response = await request.send();
    final payload = await response.stream.bytesToString();
    _throwIfFailed(response.statusCode, payload);
    return MediaAsset.fromJson(jsonDecode(payload) as Map<String, dynamic>);
  }

  Future<AnalysisSummary> analyzeMedia({
    required String mediaId,
    required String beatMethod,
    required int subdivisions,
  }) async {
    final response = await http.post(
      _uri('/api/media/$mediaId/analysis'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'beat_method': beatMethod,
        'subdivisions': subdivisions,
      }),
    );
    _throwIfFailed(response.statusCode, response.body);
    return AnalysisSummary.fromJson(jsonDecode(response.body) as Map<String, dynamic>);
  }

  Future<String> generatePracticeClip({
    required String mediaId,
    required double startSec,
    required double endSec,
    required double speedMultiplier,
    required bool mirror,
  }) async {
    final response = await http.post(
      _uri('/api/media/$mediaId/practice-clip'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'start_sec': startSec,
        'end_sec': endSec,
        'speed_multiplier': speedMultiplier,
        'mirror': mirror,
      }),
    );
    _throwIfFailed(response.statusCode, response.body);
    final payload = jsonDecode(response.body) as Map<String, dynamic>;
    return '$baseUrl${payload['preview_url']}';
  }

  Future<PoseResult> analyzePose({
    required String mediaId,
    int? sampleCount,
    List<double>? timestamps,
  }) async {
    final Map<String, dynamic> body = {};
    if (timestamps != null && timestamps.isNotEmpty) {
      body['timestamps'] = timestamps;
    } else {
      body['sample_count'] = sampleCount ?? 4;
    }
    final response = await http.post(
      _uri('/api/media/$mediaId/pose'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(body),
    );
    _throwIfFailed(response.statusCode, response.body);
    return PoseResult.fromJson(jsonDecode(response.body) as Map<String, dynamic>);
  }

  void _throwIfFailed(int statusCode, String payload) {
    if (statusCode >= 200 && statusCode < 300) {
      return;
    }
    try {
      final decoded = jsonDecode(payload) as Map<String, dynamic>;
      throw Exception(decoded['detail'] ?? payload);
    } catch (_) {
      throw Exception(payload);
    }
  }
}
