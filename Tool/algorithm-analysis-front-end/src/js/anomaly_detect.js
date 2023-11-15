import {specialAlgorithms} from './config'


const judge = function (anomaly_list, algo_score, algorithm_name, threshold) {
    let begin = -1, end = -1
    for (let i = 0; i < algo_score.length; i++) {
        if ((parseFloat(algo_score[i]) <= parseFloat(threshold) && specialAlgorithms.has(algorithm_name)) ||
            parseFloat(algo_score[i]) >= parseFloat(threshold) && !specialAlgorithms.has(algorithm_name)) {
            if (begin === -1) {
                begin = end = i
            } else {
                end = i
            }
        } else if (begin !== -1) {
            for (let j = anomaly_list.length - 1; j >= 0; j--) {
                const item = anomaly_list[j]
                if ((item[0] <= begin && begin <= item[1]) || (item[0] <= end && end <= item[1])) {
                    anomaly_list.splice(j, 1)
                }
            }
            begin = end = -1
        }
    }
    if (begin !== -1) {
        for (let j = anomaly_list.length - 1; j >= 0; j--) {
            const item = anomaly_list[j]
            if (item[0] <= begin <= item[1] || item[0] <= end <= item[1]) {
                anomaly_list.splice(j, 1)
            }
        }
    }
}

const extract_anomaly = function (label) {
    const anomaly_list = []
    let begin = -1, end = -1
    for (let i = 0; i < label.length; i++) {
        if (parseInt(label[i]) === 1) {
            if (begin === -1) {
                begin = end = i
            } else {
                end = i
            }
        } else if (begin !== -1) {
            anomaly_list.push([begin, end])
            begin = end = -1
        }
    }
    if (begin !== -1) {
        anomaly_list.push([begin, end])
    }
    return anomaly_list
}

const detection = function (label, rightData) {
    let anomaly_list = extract_anomaly(label)
    for (let i = 0; i < rightData.length; i++) {
        judge(anomaly_list, rightData[i]['score'], rightData[i]['title'], rightData[i]['threshold'])
    }
    return anomaly_list
}

export {detection}
