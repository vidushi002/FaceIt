import React, {Component} from 'react';
import Webcam from "react-webcam";
import { connect } from 'react-redux';
import axios from 'axios' 
import './App.css';
import logo from './logo.jpeg';

class Webcap extends Component {

    constructor(props){
        super(props)
        this.state = {
            imageData: null,
            image_name: "",
            saveimage: false,
            class: null,
            probability: null,
            predFlag: true,
            timeInSec:0,
            color:null,
            showPrediction:false
    
        }
    }
    
    setRef = (webcam) => {
        this.webcam = webcam;}

    capture = () => {
        let interval = null;
        this.setState({
            predFlag:true,
            showPrediction:true,
        })

        if (this.state.predFlag) {
        interval = setInterval(() => {
            if (this.state.predFlag) {
                const imageSrc = this.webcam.getScreenshot();
                this.setState({
                    imageData : imageSrc,
                    timeInSec: this.state.timeInSec+1})
                axios.post('http://127.0.0.1:5000/getPrediction',this.state).
                then(response => {
                    console.log(response)
                    let value=response.data.class
                    this.setState({
                        class: value,
                        showPrediction:true,
                        color: value ==1  ? "green":"red" ,
                    })
                })
                .catch(error =>{
                    console.log(error)})

          }}, 1000);
        }
        
    };


    stopcapture = () => {

        this.setState({
            predFlag : false,
            showPrediction:false
        })
        
    };
    // function Prediction() {
    //     if (this.state.predFlag) {
    //       return <button> {this.state.class == 1 ? "Mask Detected":"No Mask Detected" }</button>;
    //     }
    //     return <button>Not predicting</button>;
    //   };


    render() {
        const videoConstraints = {
            width: 480,
            height: 360,
            facingMode: 'user',
        };
        return (
            <div style={{backgroundColor:'white'}} >
                <div
                style={{
                    display: "flex",
                    justifyContent: "center",
                    alignItems: "center"
                }}>
               <img className="photo" src={logo} alt="Logo" />
               </div>
                <div
                style={{
                    display: "flex",
                    justifyContent: "center",
                    alignItems: "center"
                }}>
                    <Webcam 
                    audio = {false}
                    screenshotFormat="image/jpeg"
                    width={560}
                    height={420}
                    ref={this.setRef}
                    videoConstraints={videoConstraints}/>
                </div>
                <div
                style={{
                    display: "flex",
                    justifyContent: "center",
                    alignItems: "center"
                }}>
                    <br></br>
                    <br></br>
                    
                    <button  style={{padding: "12px 28px"}} onClick={() => this.capture()}>Start Predicting</button>
                    <button style={{padding: "12px 28px"}} onClick={() => this.stopcapture()}>Stop Predicting</button>
                    <br></br>
                    <br></br>
                    <br></br>
                </div>
                {this.state.showPrediction &&
                <div
                style={{
                    display: "flex",
                    justifyContent: "center",
                    alignItems: "center",
                    color:this.state.color,
                    fontSize:"30px"
                    //  
                }}>
                  { this.state.class == 1 ? "Mask Detected":"No Mask Detected" }
            </div>   }
            </div>
            
        )
    }

}

export default Webcap;