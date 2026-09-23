"""Mock session data — edit here for local / demo enrollment + playback."""

# timestamp = 当前播放进度
# user_courses = 选课清单 + 当前课/当前讲
# 材料目录: data/lectures/<course_id>/<lecture_id>/{doc.txt, transcript.vtt, meta.json}
MOCK_SESSION = {
    "timestamp": "01:22:09",
    "user_courses": {
        "current_quarter": "2026-Spring",
        "current_course": "CSC447",
        "current_lecture": "lec01",
        "courses": [
            {
                "course_id": "CSC447",
                "quarter": "2026-Spring",
                "lecturer": "Eric J. Fredericks",
                "lecture_id": ["lec01"],
            },
            {
                "course_id": "CSC477",
                "quarter": "2026-Spring",
                "lecturer": "Eric J. Fredericks",
                "lecture_id": [
                    "lec01",
                    "lec02",
                    "lec03",
                    "lec04",
                    "lec05",
                    "lec06",
                    "lec07",
                    "lec08",
                    "lec09",
                    "lec10",
                ],
            },
            {
                "course_id": "CSC421",
                "quarter": "2025-Fall",
                "lecturer": "Unknown",
                "lecture_id": ["lec01"],
            },
            {
                "course_id": "CSC435",
                "quarter": "2025-Fall",
                "lecturer": "Unknown",
                "lecture_id": ["lec02"],
            },
        ],
    },
}

# 便捷别名（只读 MOCK_SESSION）
TIMESTAMP = MOCK_SESSION["timestamp"]
USER_COURSES = MOCK_SESSION["user_courses"]
CURRENT_QUARTER = USER_COURSES["current_quarter"]
CURRENT_COURSE = USER_COURSES["current_course"]
CURRENT_LECTURE = USER_COURSES["current_lecture"]

# ingest / 无 course_ctx 时的默认视频最长时长
DEFAULT_LECTURE_MAX_TS = "03:30:00"
