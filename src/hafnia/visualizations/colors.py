from typing import List, Tuple


def get_n_colors(index: int) -> List[Tuple[int, int, int]]:
    n_colors = len(COLORS)
    colors = [COLORS[index % n_colors] for index in range(index)]
    return colors


COLORS = [
    (210, 24, 32),
    (24, 105, 255),
    (0, 138, 0),
    (243, 109, 255),
    (113, 0, 121),
    (170, 251, 0),
    (0, 190, 194),
    (255, 162, 53),
    (93, 61, 4),
    (8, 0, 138),
    (0, 93, 93),
    (154, 125, 130),
    (162, 174, 255),
    (150, 182, 117),
    (158, 40, 255),
    (77, 0, 20),
    (255, 174, 190),
    (206, 0, 146),
    (0, 255, 182),
    (0, 45, 0),
    (158, 117, 0),
    (61, 53, 65),
    (243, 235, 146),
    (101, 97, 138),
    (138, 61, 77),
    (89, 4, 186),
    (85, 138, 113),
    (178, 190, 194),
    (255, 93, 130),
    (28, 198, 0),
    (146, 247, 255),
    (45, 134, 166),
    (57, 93, 40),
    (235, 206, 255),
    (255, 93, 0),
    (166, 97, 170),
    (134, 0, 0),
    (53, 0, 89),
    (0, 81, 142),
    (158, 73, 16),
    (206, 190, 0),
    (0, 40, 40),
    (0, 178, 255),
    (202, 166, 134),
    (190, 154, 194),
    (45, 32, 12),
    (117, 101, 69),
    (130, 121, 223),
    (0, 194, 138),
    (186, 231, 194),
    (134, 142, 166),
    (202, 113, 89),
    (130, 154, 0),
    (45, 0, 255),
    (210, 4, 247),
    (255, 215, 190),
    (146, 206, 247),
    (186, 93, 125),
    (255, 65, 194),
    (190, 134, 255),
    (146, 142, 101),
    (166, 4, 170),
    (134, 227, 117),
    (73, 0, 61),
    (251, 239, 12),
    (105, 85, 93),
    (89, 49, 45),
    (105, 53, 255),
    (182, 4, 77),
    (93, 109, 113),
    (65, 69, 53),
    (101, 113, 0),
    (121, 0, 73),
    (28, 49, 81),
    (121, 65, 158),
    (255, 146, 113),
    (255, 166, 243),
    (186, 158, 65),
    (130, 170, 154),
    (215, 121, 0),
    (73, 61, 113),
    (81, 162, 85),
    (231, 130, 182),
    (210, 227, 251),
    (0, 73, 49),
    (109, 219, 194),
    (61, 77, 93),
    (97, 53, 85),
    (0, 113, 81),
    (93, 24, 0),
    (154, 93, 81),
    (85, 142, 219),
    (202, 202, 154),
    (53, 24, 32),
    (57, 61, 0),
    (0, 154, 150),
    (235, 16, 109),
    (138, 69, 121),
    (117, 170, 194),
    (202, 146, 154),
    (210, 186, 198),
    (154, 206, 0),
    (69, 109, 170),
    (117, 89, 0),
    (206, 77, 12),
    (0, 223, 251),
    (255, 61, 65),
    (255, 202, 73),
    (45, 49, 146),
    (134, 105, 134),
    (158, 130, 190),
    (206, 174, 255),
    (121, 69, 45),
    (198, 251, 130),
    (93, 117, 73),
    (182, 69, 73),
    (255, 223, 239),
    (162, 0, 113),
    (77, 77, 166),
    (166, 170, 202),
    (113, 28, 40),
    (40, 121, 121),
    (8, 73, 0),
    (0, 105, 134),
    (166, 117, 73),
    (251, 182, 130),
    (85, 24, 125),
    (0, 255, 89),
    (0, 65, 77),
    (109, 142, 146),
    (170, 36, 0),
    (190, 210, 109),
    (138, 97, 186),
    (210, 65, 190),
    (73, 97, 81),
    (206, 243, 239),
    (97, 194, 97),
    (20, 138, 77),
    (0, 255, 231),
    (0, 105, 0),
    (178, 121, 158),
    (170, 178, 158),
    (186, 85, 255),
    (198, 121, 206),
    (32, 49, 32),
    (125, 4, 219),
    (194, 198, 247),
    (138, 198, 206),
    (231, 235, 206),
    (40, 28, 57),
    (158, 255, 174),
    (130, 206, 154),
    (49, 166, 12),
    (0, 162, 117),
    (219, 146, 85),
    (61, 20, 4),
    (255, 138, 154),
    (130, 134, 53),
    (105, 77, 113),
    (182, 97, 0),
    (125, 45, 0),
    (162, 178, 57),
    (49, 4, 125),
    (166, 61, 202),
    (154, 32, 45),
    (4, 223, 134),
    (117, 125, 109),
    (138, 150, 210),
    (8, 162, 202),
    (247, 109, 93),
    (16, 85, 202),
    (219, 182, 101),
    (146, 89, 109),
    (162, 255, 227),
    (89, 85, 40),
    (113, 121, 170),
    (215, 89, 101),
    (73, 32, 81),
    (223, 77, 146),
    (0, 0, 202),
    (93, 101, 210),
    (223, 166, 0),
    (178, 73, 146),
    (182, 138, 117),
    (97, 77, 61),
    (166, 150, 162),
    (85, 28, 53),
    (49, 65, 65),
    (117, 117, 134),
    (146, 158, 162),
    (117, 154, 113),
    (255, 130, 32),
    (134, 85, 255),
    (154, 198, 182),
    (223, 150, 243),
    (202, 223, 49),
    (142, 93, 40),
    (53, 190, 227),
    (113, 166, 255),
    (89, 138, 49),
    (255, 194, 235),
    (170, 61, 105),
    (73, 97, 125),
    (73, 53, 28),
    (69, 178, 158),
    (28, 36, 49),
    (247, 49, 239),
    (117, 0, 166),
    (231, 182, 170),
    (130, 105, 101),
    (227, 162, 202),
    (32, 36, 0),
    (121, 182, 16),
    (158, 142, 255),
    (210, 117, 138),
    (202, 182, 219),
    (174, 154, 223),
    (255, 113, 219),
    (210, 247, 178),
    (198, 215, 206),
    (255, 210, 138),
    (93, 223, 53),
    (93, 121, 146),
    (162, 142, 0),
    (174, 223, 239),
    (113, 77, 194),
    (125, 69, 0),
    (101, 146, 182),
    (93, 121, 255),
    (81, 73, 89),
    (150, 158, 81),
    (206, 105, 174),
    (101, 53, 117),
    (219, 210, 227),
    (182, 174, 117),
    (81, 89, 0),
    (182, 89, 57),
    (85, 4, 235),
    (61, 117, 45),
    (146, 130, 154),
    (130, 36, 105),
    (186, 134, 57),
    (138, 178, 227),
    (109, 178, 130),
    (150, 65, 53),
    (109, 65, 73),
    (138, 117, 61),
    (178, 113, 117),
    (146, 28, 73),
    (223, 109, 49),
    (0, 227, 223),
    (146, 4, 202),
    (49, 40, 89),
    (0, 125, 210),
    (162, 109, 255),
    (130, 89, 146),
]
