const jsonFiles = [
    'mars-science-laboratory.json',
    'water-distribution.json',
    'soil-moisture-active-passive.json',
    'server-machine-dataset.json',
    'secure-water-treatment.json',
    'application-server-dataset.json',
    'yidong-38.json',
    'yidong-22.json',
    'CTF.json',
    'ASD-RESID.json'
]

const specialAlgorithms = new Set([
    'OmniAnomaly',
    'sdfvae',
    'DOMI',
    'InterFusion',
    'SDFVAE'
])

const backgroundColorLevel = [
    "rgb(0,0,0)",
    "rgb(232,159,159)",
    "rgb(236,80,80)",
    "rgb(239,42,42)",
    "rgb(255,0,0)",
]


const showQuotas = {
    'server-machine-dataset': [
        0, 1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 22, 23, 24, 26,
        28, 29, 31, 32, 33, 34,
    ],
    'soil-moisture-active-passive': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 19, 21,
    ],
    'mars-science-laboratory': [
        0, 33, 5, 6, 39, 41, 11, 12, 43, 47, 19, 20, 53, 54, 27, 29, 31,
    ],
    'secure-water-treatment': [
        0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14, 16, 17, 18, 19, 20, 21, 22, 25, 26,
        27, 28, 29, 31, 34, 35, 36, 37, 43, 45, 47, 49, 50,
    ],
    'water-distribution': [
        0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 13, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 47, 48, 49, 50, 51, 52,
        53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 70, 73, 75, 76,
        77, 78, 79, 80, 81, 83, 85, 86, 87, 88, 97, 98, 99, 100, 101, 102, 103, 104,
        106, 107, 108, 109, 112, 120, 122,
    ],
    'application-server-dataset': [0, 1, 2, 3, 4, 5, 6, 7, 8, 18],
    'yidong-22': [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 13, 14, 19],
    'yidong-38': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14, 15, 17, 19, 21, 22, 23, 25, 27, 29,
        33, 36,
    ],
    'CTF': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43, 44, 45, 46, 47, 48],
    'ASD-RESID': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
}
export {jsonFiles, specialAlgorithms, showQuotas, backgroundColorLevel}
