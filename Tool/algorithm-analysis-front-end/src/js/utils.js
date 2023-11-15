import { specialAlgorithms } from './config'

const getCSVDataForm = (data, timeTrack, label, algorithm, threshold) => {
  const results = []
  var min = Number.MAX_SAFE_INTEGER,
    max = Number.MIN_SAFE_INTEGER
  if (threshold) {
    results.push([timeTrack[0], 0, 0, 0])
  } else {
    results.push([timeTrack[0], 0, 0])
  }
  var abnormal = false
  for (let i = 0, time = timeTrack[0]; i < data.length; i++, time++) {
    let pointStatus
    if (label) {
      pointStatus = label[i]
    } else {
      if (specialAlgorithms.has(algorithm)) {
        pointStatus = data[i] <= threshold ? 1 : 0
      } else {
        pointStatus = data[i] >= threshold ? 1 : 0
      }
    }
    if (abnormal) {
      if (pointStatus === 0) {
        if (threshold) {
          results.push([time + 0.5, threshold, threshold, threshold])
          results.push([time + 1, data[i], null, threshold])
        } else {
          results.push([time + 1, data[i], data[i]])
        }
        abnormal = false
      } else {
        if (threshold) {
          results.push([time + 1, null, data[i], threshold])
        } else {
          results.push([time + 1, null, data[i]])
        }
      }
    } else {
      if (pointStatus === 0) {
        if (threshold) {
          results.push([time + 1, data[i], null, threshold])
        } else {
          results.push([time + 1, data[i], null])
        }
      } else {
        if (threshold) {
          results.push([time + 0.5, threshold, threshold, threshold])
          results.push([time + 1, null, data[i], threshold])
        } else {
          results.push([time + 1, data[i], data[i]])
        }
        abnormal = true
      }
    }
    min = Math.min(min, data[i])
    max = Math.max(max, data[i])
  }
  if (threshold) {
    min = Math.min(min, threshold)
    max = Math.max(max, threshold)
  }
  return [results, max, min]
}

const extractXAxis = (yValue, threshold, title) => {
  let extractXAxis = []
  let begin = -1
  let end = -1
  if (!specialAlgorithms.has(title)) {
    for (let i = 0; i < yValue.length; i++) {
      if (yValue[i] >= threshold) {
        if (begin === -1) {
          begin = i
          end = begin
        } else {
          end = i
        }
      } else {
        if (begin !== -1) {
          if (begin !== end) {
            extractXAxis.push(begin)
            extractXAxis.push(end)
          } else {
            extractXAxis.push(begin)
            extractXAxis.push(begin + 1)
          }
          begin = -1
          end = -1
        }
      }
    }
  } else {
    for (let i = 0; i < yValue.length; i++) {
      if (yValue[i] <= threshold) {
        if (begin === -1) {
          begin = i
          end = begin
        } else {
          end = i
        }
      } else {
        if (begin !== -1) {
          if (begin !== end) {
            extractXAxis.push(begin)
            extractXAxis.push(end)
          } else {
            extractXAxis.push(begin)
            extractXAxis.push(begin + 1)
          }
          begin = -1
          end = -1
        }
      }
    }
  }
  if (begin !== -1) {
    if (begin !== end) {
      extractXAxis.push(begin)
      extractXAxis.push(end)
    } else {
      extractXAxis.push(begin)
      extractXAxis.push(begin + 1)
    }
  }
  return extractXAxis
}

export { getCSVDataForm, extractXAxis }
