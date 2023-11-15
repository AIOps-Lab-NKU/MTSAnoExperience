<template>
  <div class="main-content">
    <template v-for="(item,index) in quotas" :key="index">
      <div class="show-echarts">
      </div>
    </template>
  </div>
</template>

<script>
import {extractXAxis, getCSVDataForm} from "../js/utils";
import {detection} from "../js/anomaly_detect";
import {showQuotas, backgroundColorLevel} from "../js/config";

export default {
  name: "showRight",
  data() {
    return {
      marked: [],
      quotas: [],
      graphics: [],
      selectArr: {},
      anomaly_list: []
    }
  },
  methods: {
    Init() {
      this.marked.length = 0;
      this.selectArr = {}
      this.marked.push({
        section: [0, this.getRightData[0].length - 1],
        depth: 0
      })
    },
    showEcharts() {
      for (let i = 0; i < this.graphics.length; i++) {
        this.graphics[i].destroy()
      }
      this.graphics.length = 0
      const showDivs = document.getElementsByClassName("show-echarts");
      for (let i = 0; i < showDivs.length; i++) {
        const [data, max, min] = getCSVDataForm(this.getRightData[i], this.$store.state.timesTrack, this.getLabels)
        const _that = this
        this.graphics.push(new this.$dygraphs(showDivs[i], data, {
          legend: "always",
          axis: {
            y: {
              valueParser: function (x) {
                return x.toPrecision(2)
              },
            }
          },
          drawGrid: false,
          valueRange: [min, max],
          includeZero: false,
          colors: ['green', 'blue'],
          // showRangeSelector: true,
          drawCallback: function (dygraph, is_initial) {
            _that.$finishLoading()
          },
          animatedZooms: true,
          underlayCallback: function (canvas, area, g) {
            function highlight_period(x_start, x_end) {
              var canvas_left_x = g.toDomXCoord(x_start);
              var canvas_right_x = g.toDomXCoord(x_end);
              var canvas_width = canvas_right_x - canvas_left_x;
              canvas.fillRect(canvas_left_x, area.y, canvas_width, area.h);
            }

            for (let j = 0; j < _that.anomaly_list.length; j++) {
              canvas.fillStyle = 'red';
              highlight_period(g.getValue(_that.anomaly_list[j][0] + 1, 0), g.getValue(_that.anomaly_list[j][1] + 1, 0))
            }
            for (let i = 0; i < _that.marked.length; i++) {
              if (_that.marked[i].depth > 0) {
                canvas.fillStyle = backgroundColorLevel[_that.marked[i].depth > backgroundColorLevel.length ? backgroundColorLevel.length - 1 : _that.marked[i].depth - 1];
                highlight_period(g.getValue(_that.marked[i].section[0] + 1, 0), g.getValue(_that.marked[i].section[1] + 1, 0))
              }
            }
          },
          legendFormatter: function (data) {
            if (data.x) {
              return `<span class="legend-div" style="position:absolute;
                        background-color: aquamarine;
                        width: fit-content;
                        height: fit-content;
                        z-index: 999;
                        top:${window.event.offsetY + 14}px;
                        left:${window.event.offsetX + 10}px">${data.x}</br>
                         指标:${_that.quotas[i]}</br>
                         数值:${data.series[0].yHTML ? data.series[0].yHTML : data.series[1].yHTML}
                        </span>`
            } else {
              return ''
            }
          }
        }))
      }
    },
    subExtractLabel(beginIndex, arr) {
      let begin = -1;
      let end = -1;
      for (let i = beginIndex; i < this.marked.length; i++) {
        if (this.marked[i].section[0] <= arr[0] && arr[0] <= this.marked[i].section[1]) {
          begin = i
        }
        if (this.marked[i].section[0] <= arr[1] && arr[1] <= this.marked[i].section[1]) {
          end = i;
          break;
        }
      }
      //在中间
      if (begin === end) {
        const temp = this.marked[begin];
        let endIndex = temp.section[1]
        temp.section[1] = arr[0]
        this.marked.splice(begin + 1, 0, {
          section: arr,
          depth: temp.depth - 1
        }, {
          section: [arr[1], endIndex],
          depth: temp.depth
        })
        this.cleanMarked(Math.max(begin - 1, 0), Math.min(begin + 4, this.marked.length - 1))
      }
      //在多测
      else {
        let endIndex = this.marked[begin].section[1]
        this.marked[begin].section[1] = arr[0]
        this.marked.splice(begin + 1, 0, {
          section: [arr[0], endIndex],
          depth: this.marked[begin].depth - 1
        })
        for (let i = begin + 2; i <= end; i++) {
          this.marked[i].depth--
        }
        this.marked.splice(end + 1, 0, {
          section: [this.marked[end + 1].section[0], arr[1]],
          depth: this.marked[end + 1].depth - 1
        })
        this.marked[end + 2].section[0] = arr[1];
        this.cleanMarked(Math.max(begin - 1, 0), Math.min(end + 4, this.marked.length - 1))
      }
      return Math.max(begin - 1, 0);
    },
    addExtractLabel(beginIndex, arr) {
      let begin = -1;
      let end = -1;
      for (let i = beginIndex; i < this.marked.length; i++) {
        if (this.marked[i].section[0] <= arr[0] && arr[0] <= this.marked[i].section[1]) {
          begin = i
        }
        if (this.marked[i].section[0] <= arr[1] && arr[1] <= this.marked[i].section[1]) {
          end = i;
          break;
        }
      }
      //在中间
      if (begin === end) {
        const temp = this.marked[begin];
        let endIndex = temp.section[1]
        temp.section[1] = arr[0]
        this.marked.splice(begin + 1, 0, {
          section: arr,
          depth: temp.depth + 1
        }, {
          section: [arr[1], endIndex],
          depth: temp.depth
        })
        this.cleanMarked(Math.max(begin - 1, 0), Math.min(begin + 4, this.marked.length - 1))
      }
      //在多测
      else {
        let endIndex = this.marked[begin].section[1]
        this.marked[begin].section[1] = arr[0]
        this.marked.splice(begin + 1, 0, {
          section: [arr[0], endIndex],
          depth: this.marked[begin].depth + 1
        })
        for (let i = begin + 2; i <= end; i++) {
          this.marked[i].depth++
        }
        this.marked.splice(end + 1, 0, {
          section: [this.marked[end + 1].section[0], arr[1]],
          depth: this.marked[end + 1].depth + 1
        })
        this.marked[end + 2].section[0] = arr[1];
        this.cleanMarked(Math.max(begin - 1, 0), Math.min(end + 4, this.marked.length - 1))
      }
      return Math.max(begin - 1, 0);
    },
    cleanMarked(beginIndex, endIndex) {
      var arr = [];
      let depth = -1;
      let begin = -1;
      let end = -1;
      for (let i = beginIndex; i <= endIndex; i++) {
        if (this.marked[i].section[0] === this.marked[i].section[1]) {
          continue
        }
        if (this.marked[i].depth === depth) {
          end = this.marked[i].section[1]
        } else {
          if (begin !== -1) {
            arr.push({
              section: [begin, end],
              depth: depth,
            })
          }
          begin = this.marked[i].section[0]
          end = this.marked[i].section[1]
          depth = this.marked[i].depth
        }
      }
      arr.push({
        section: [begin, end],
        depth: depth,
      })
      this.marked.splice(beginIndex, endIndex - beginIndex + 1)
      for (let i = beginIndex; i < beginIndex + arr.length; i++) {
        this.marked.splice(i, 0, arr[i - beginIndex]);
      }
    },
    changeMap() {
      this.showEcharts()
      this.$nextTick(() => {
        this.$finishLoading();
      })
    }
  },
  computed: {
    getDataSet() {
      return this.$store.state.datasetName
    },
    getSelectLength() {
      return this.$store.state.selected.size
    },
    getIsClear() {
      return this.$store.state.isCleared
    },
    getRightData() {
      return this.$store.state.rightData
    },
    getLabels() {
      return this.$store.state.labels
    }
  },
  watch: {
    getSelectLength: {
      deep: true,
      handler(newValue, oldValue) {
        if (this.$store.state.isCleared) {
          return
        }
        if (newValue > oldValue) {
          const a = this.$store.state.lastChangeSelect
          let exs;
          if (this.selectArr.hasOwnProperty(a)) {
            exs = this.selectArr[a]
          } else {
            exs = extractXAxis(this.$store.state.middleData[a]['score'], this.$store.state.middleData[a]['threshold'], this.$store.state.middleData[a]['title']);
            this.selectArr[a] = exs;
          }
          let begin = 0;
          for (let i = 0; i < exs.length - 1; i += 2) {
            begin = this.addExtractLabel(begin, [exs[i], exs[i + 1]])
          }
          this.changeMap()
        } else if (newValue < oldValue) {
          const a = this.$store.state.lastChangeSelect
          const exs = this.selectArr[a]
          let begin = 0;
          for (let i = 0; i < exs.length - 1; i += 2) {
            begin = this.subExtractLabel(begin, [exs[i], exs[i + 1]])
          }
          this.changeMap()
        }
      },
    },
    getRightData: {
      deep: true,
      handler(value) {
        this.quotas.length = 0
        if (this.$store.state.isShowAll) {
          for (let i = 0; i < value.length; i++) {
            this.quotas.push(i);
          }
        } else {
          for (let i = 0; i < showQuotas[this.getDataSet].length; i++) {
            this.quotas.push(showQuotas[this.getDataSet][i]);
          }
        }
        this.anomaly_list.length = 0
        this.anomaly_list = detection(this.getLabels, this.$store.state.middleData)
        this.$nextTick(() => {
          this.Init()
          this.showEcharts();
        })
      },
    },
  }
}
</script>

<style scoped>
.main-content {
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.show-echarts {
  width: 99%;
  height: 180px;
  margin: auto;
  margin-left: -20px;
  /*background-color: #1883e0;*/
}
</style>
