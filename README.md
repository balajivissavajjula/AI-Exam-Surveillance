\# AI Exam Surveillance System



An AI-powered smart examination hall surveillance and malpractice detection

system designed to monitor examination environments using computer vision,

gaze estimation, face identity verification, behavioral feature extraction,

and machine learning-based classification.



\---



\## Overview



The AI Exam Surveillance System processes CCTV/IP camera video frames and

combines multiple AI components to identify examination-related objects,

analyze student behavior, estimate gaze direction, verify student identity,

and classify observed behavior as either \*\*Normal\*\* or \*\*Cheating\*\*.



The system is designed around the following pipeline:



```text

CCTV / IP Camera

&#x20;       |

&#x20;       v

&#x20;    OpenCV

&#x20;       |

&#x20;       v

&#x20;    YOLO11m

&#x20;       |

&#x20;       +----------------------+

&#x20;       |                      |

&#x20;       v                      v

&#x20;  Object Detection       MediaPipe

&#x20;                         Face / Eye /

&#x20;                         Hand Landmarks

&#x20;                             |

&#x20;                             v

&#x20;                         Head Pose

&#x20;                             |

&#x20;                             v

&#x20;                        GazeNet V2

&#x20;                             |

&#x20;                             v

&#x20;                   Face Identity Module

&#x20;                             |

&#x20;                             v

&#x20;                   37-Feature Extraction

&#x20;                             |

&#x20;                             v

&#x20;                        XGBoost

&#x20;                             |

&#x20;                   +---------+---------+

&#x20;                   |                   |

&#x20;                 Normal             Cheating

&#x20;                   |                   |

&#x20;                   +---------+---------+

&#x20;                             |

&#x20;                   +---------+---------+

&#x20;                   |                   |

&#x20;                   v                   v

&#x20;                Firebase             ESP32

&#x20;             Event Logging       Buzzer / LED

&#x20;                   |

&#x20;                   v

&#x20;            Administrator Dashboard

