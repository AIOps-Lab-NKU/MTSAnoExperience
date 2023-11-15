<template>
  <div class="left">
    <div class="text-center top">
      <div @click="displayBasic = true">
        {{ selectedFile === null ? '选择数据集合' : selectedFile }}
      </div>
    </div>
    <div>
      是否展示所有指标：
      <el-switch
          v-model="isShowAll"
          class="ml-2"
          inline-prompt
          style="--el-switch-on-color: #13ce66; --el-switch-off-color: #ff4949"
          active-text="是"
          inactive-text="否"
      />
    </div>
    <div style="margin-top: 10px">
      数据节点总个数：{{ totalTimeTrack }}
    </div>
    <div style="margin-top: 10px">
      观察的时间节点数：
      <el-input-number v-model="length" :min="2" :max="Math.min(maxLength,totalTimeTrack - beginValue)"
                       @change="changeData"/>
    </div>
    <div style="margin-top: 10px">
      起始时间节点：
      <el-input-number v-model="beginValue" :min="0" :max="totalTimeTrack-length"
                       :step="step"
                       @change="changeData"/>
    </div>
    <div style="margin-top:30px; display: flex;flex-direction: column">
      <template v-for="(item,index) in Object.keys(allData)" :key="index">
        <el-radio v-model="dataSetItem" :label="item" size="large" border @change="changeDataSet(dataSetItem)">
          {{ item }}
        </el-radio>
      </template>
    </div>
    <el-dialog
        v-model="displayBasic"
        title="选择数据集"
        width="30%">
      <template v-for="(jsonFile,index) in jsonFiles" :key="index">
        <el-radio v-model="selectedFile" :label="jsonFile" size="large" border>
          {{ jsonFile }}
        </el-radio>
      </template>
      <template #footer>
      <span class="dialog-footer">
        <el-button type="primary" @click="readFile">确定</el-button>
      </span>
      </template>
    </el-dialog>
  </div>
</template>

<script>
import {jsonFiles, showQuotas} from "../js/config";
import {ElMessage} from 'element-plus'

export default {
  name: 'leftBar',
  data() {
    return {
      jsonFiles: null,
      selectedFile: null,
      displayBasic: false,
      beginValue: 0,
      maxLength: 20000,
      totalTimeTrack: 20000,
      dataSetItem: null,
      length: 20000,
      allData: {},
      step: 20000,
      isShowAll: false,
    }
  },
  methods: {
    readFile() {
      this.displayBasic = false
      this.$showLoading()
      this.$http.get('/' + this.selectedFile)
          .then((res) => {
            this.allData = res.data
            this.dataSetItem = Object.keys(this.allData)[0]
            this.changeDataSet(this.dataSetItem)
            this.$finishLoading()
          })
          .catch((err) => {
            console.log(err)
            this.$finishLoading()
            ElMessage({
              showClose: true,
              message: '文件过大！！',
              type: 'error',
            })
          })
    },
    changeData() {
      if (this.dataSetItem === null) {
        ElMessage({
          showClose: true,
          message: '请选择数据！！',
          type: 'error',
        })
        return
      }
      this.$showLoading()
      const datasetName = this.selectedFile.slice(0, this.selectedFile.indexOf('.'))
      setTimeout(() => {
        this.$store.commit('clearConstant')
        this.$store.commit('setDatasetName', datasetName)
        this.$store.commit('setTimeTrack', [this.beginValue, this.length])
        const algorithms = Object.keys(this.allData[this.dataSetItem]).slice(2)
        const middleData = []
        for (let i = 0; i < algorithms.length; i++) {
          let temp = {};
          temp['threshold'] = this.allData[this.dataSetItem][algorithms[i]]['threshold']
          temp['title'] = algorithms[i];
          temp['score'] = this.allData[this.dataSetItem][algorithms[i]]['score'].slice(this.beginValue, this.beginValue + this.length)
          temp['p'] = this.allData[this.dataSetItem][algorithms[i]]['p']
          temp['r'] = this.allData[this.dataSetItem][algorithms[i]]['r']
          temp['f1'] = this.allData[this.dataSetItem][algorithms[i]]['f1']
          middleData.push(temp)
        }
        this.$store.commit('setMiddleData', middleData)
        const rightData = []
        if (this.isShowAll) {
          for (let i = 0; i < this.allData[this.dataSetItem]['data'].length; i++) {
            rightData.push(this.allData[this.dataSetItem]['data'][i].slice(this.beginValue, this.beginValue + this.length))
          }
        } else {
          for (let i in showQuotas[datasetName]) {
            rightData.push(this.allData[this.dataSetItem]['data'][showQuotas[datasetName][i]].slice(this.beginValue, this.beginValue + this.length))
          }
        }
        this.$store.commit('setRightData', rightData)
        this.$store.commit('setLabels', this.allData[this.dataSetItem]['label'].slice(this.beginValue, this.beginValue + this.length))
      }, 250)
    },
    changeDataSet(dataSetItem) {
      this.totalTimeTrack = this.allData[dataSetItem]['label'].length
      this.length = Math.min(this.totalTimeTrack, this.maxLength)
      this.beginValue = 0
      this.changeStep()
      this.changeData()
    },
    changeStep() {
      this.step = this.length
      if ((this.beginValue + this.step + this.length > this.totalTimeTrack)
          && (this.totalTimeTrack - this.beginValue - this.length > 0)) {
        this.step = this.totalTimeTrack - this.beginValue - this.length
      }
    }
  },
  created() {
    this.jsonFiles = jsonFiles
  },
  watch: {
    beginValue(newValue, oldValue) {
      if (newValue > oldValue && newValue < this.totalTimeTrack - this.length) {
        this.changeStep()
      } else {
        this.step = Math.min(newValue, this.length);
        if (this.step === 0) {
          this.changeData()
        }
      }
    },
    length(newValue, oldValue) {
      this.changeStep()
    },
    isShowAll(newValue, oldValue) {
      if (oldValue !== newValue) {
        this.$store.commit('setShowAll', this.isShowAll)
        this.$showLoading()
        setTimeout(() => {
          if (!this.allData || !this.selectedFile) {
            this.$finishLoading()
            return
          }
          const datasetName = this.selectedFile.slice(0, this.selectedFile.indexOf('.'))
          const rightData = []
          if (this.isShowAll) {
            for (let i = 0; i < this.allData[this.dataSetItem]['data'].length; i++) {
              rightData.push(this.allData[this.dataSetItem]['data'][i].slice(this.beginValue, this.beginValue + this.length))
            }
          } else {
            for (let i in showQuotas[datasetName]) {
              rightData.push(this.allData[this.dataSetItem]['data'][showQuotas[datasetName][i]].slice(this.beginValue, this.beginValue + this.length))
            }
          }
          this.$store.commit('setRightData', rightData)
        }, 200)
      }
    }
  }
}
</script>

<style scoped>
.text-center {
  line-height: 100%;
  text-align: center;
}

.top {
  margin-top: 30px;
  margin-bottom: 20px;
  font-size: x-large;
  font-family: 'Al Tarikh';
}

.top:hover {
  cursor: pointer;
}

.left {
  display: flex;
  flex-direction: column;
}

.el-radio {
  margin-bottom: 15px;
  width: 80%;
}
</style>
