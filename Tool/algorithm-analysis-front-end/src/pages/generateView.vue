<template>
  <div>
    <div class="head">
      <div>
        指标数量：
        <el-input v-model="metrics" type="number" min="2" max="41"></el-input>
      </div>
      <div>
        数据长度（min）：
        <el-input v-model="time_span" type="number" readonly></el-input>
      </div>
      <div>
        采样间隔：
        <el-input v-model="sample_freq" type="number" min="2" max="15"></el-input>
      </div>
      <div>
        指标相似性比例：
        <el-input v-model="metric_similiarity_per" placeholder="0、0.1、0.2..，0.9"></el-input>
      </div>
      <div>
        周期性指标比例：
        <el-input v-model="metric_cyclicity_per" placeholder="0、0.1、0.2...，1"></el-input>
      </div>
      <div>
        噪声比例：
        <el-input v-model="noise_amp" placeholder="0、0.01、0.02..，0.1"></el-input>
      </div>
      <div>
        选项：
        <el-input v-model="time_index" type="number" min="1" max="5"></el-input>
      </div>
      <el-button type="primary" @click="load">加载</el-button>
    </div>
    <div class="main-content">
      <template v-for="index in this.data.length " :key="index">
        <div class="show-echarts">
        </div>
      </template>
    </div>
  </div>
</template>

<script>
import {ElMessage} from "element-plus";
import {getCSVDataForm} from "../js/utils";
import {backgroundColorLevel} from "../js/config";

export default {
  name: "generateView",
  data() {
    return {
      metrics: 5,
      time_span: 45 * 24 * 60,
      sample_freq: 1,
      metric_similiarity_per: 0.5,
      metric_cyclicity_per: 0.8,
      noise_amp: 0.05,
      graphics: [],
      data: [],
      label: [],
      time_index: 1,
    }
  },
  methods: {
    load() {
      this.$showLoading()
      this.$http.get(`/data/data_${this.metrics}_${this.time_span}_${this.sample_freq}_${this.metric_similiarity_per}_${this.metric_cyclicity_per}_${this.noise_amp}_${this.time_index}.csv`)
          .then((res) => {
            const resData = res.data.split('\n')
            resData.shift()
            const nums = resData[0].split(',').length - 1
            this.data.length = 0
            this.label.length = 0
            for (let i = 0; i < nums; i++) {
              this.data.push([])
            }
            for (let i = 0; i < resData.length; i++) {
              const item = resData[i].split(',')
              if (item.length === 0) {
                continue
              }
              for (let j = 0; j < nums; j++) {
                this.data[j].push(parseFloat(item[j]))
              }
              this.label.push(parseInt(item[nums]))
            }
            this.$nextTick(function () {
              this.showEcharts(this.data, this.label)
            });
          })
          .catch((err) => {
            console.log(err)
            this.$finishLoading()
            ElMessage({
              showClose: true,
              message: '文件未找到！',
              type: 'error',
            })
          })
    },
    showEcharts() {
      for (let i = 0; i < this.graphics.length; i++) {
        this.graphics[i].destroy()
      }
      const time = []
      for (let i = 0; i < this.label.length; i++) {
        time.push(i + 1)
      }
      this.graphics.length = 0
      const showDivs = document.getElementsByClassName("show-echarts");
      for (let i = 0; i < showDivs.length; i++) {
        const [data, max, min] = getCSVDataForm(this.data[i], time, this.label)
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
          stepPlot: true,
          valueRange: [min, max],
          includeZero: false,
          colors: ['green', 'blue'],
          // showRangeSelector: true,
          drawCallback: function (dygraph, is_initial) {
            _that.$finishLoading()
          },
          animatedZooms: true,
          legendFormatter: function (data) {
            if (data.x) {
              return `<span class="legend-div" style="position:absolute;
                        background-color: aquamarine;
                        width: fit-content;
                        height: fit-content;
                        z-index: 999;
                        top:${window.event.offsetY + 14}px;
                        left:${window.event.offsetX + 10}px">${data.x}</br>
                         指标:${i}</br>
                         数值:${data.series[0].yHTML ? data.series[0].yHTML : data.series[1].yHTML}
                        </span>`
            } else {
              return ''
            }
          }
        }))
      }
    }
  }
}
</script>

<style scoped>
.head {
  display: flex;
  width: 80%;
  text-align: center;
  margin: auto;
}

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
