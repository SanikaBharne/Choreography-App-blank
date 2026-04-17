class MediaAsset {
  const MediaAsset({
    required this.mediaId,
    required this.filename,
    required this.mediaType,
    required this.originalUrl,
  });

  final String mediaId;
  final String filename;
  final String mediaType;
  final String originalUrl;

  factory MediaAsset.fromJson(Map<String, dynamic> json) {
    return MediaAsset(
      mediaId: json['media_id'] as String,
      filename: json['filename'] as String,
      mediaType: json['media_type'] as String,
      originalUrl: json['original_url'] as String,
    );
  }
}
