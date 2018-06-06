import React, { Component } from 'react'
import Particles from 'react-particles-js'
import './App.css'
class App extends Component {
  render () {
    return (
      <div className='content-wrapper'>
        <div class='sidebar'>

          <div className='profile'>
            <div className='profile-text'>
              <div className='title'>
                KAUNIL DHRUV
              </div>
              <div className='subtitle'><span>Software Engineer</span></div>
            </div>

            <div className='social-links'>
              <ul>
                <li>
                  <img src='assets/github.png' />
                </li>
                <li>
                  <img src='assets/linkedin.png' />
                </li>
                <li>
                  <img src='assets/scholar.png' />
                </li>
              </ul>
            </div>
          </div>

          <Particles
            params={{
              particles: {
                line_linked: {
                  shadow: {
                    enable: true,
                    color: '#3CA9D1',
                    blur: 2
                  }
                }
              }
            }}
            canvasClassName='particles'
          />
        </div>
        <div class='content'>

          <div className='page-1'>
            <div className='page-content'>
              <div className='page-content-inner'>
                <div className='page-title'>
                  <p>
                    Biography
                  </p>
                </div>
                <div className='page-seperator show' />
                <div className='page-descp'>
                  <p>
                  I joined <a target='blank' className='facebook' href='https://research.fb.com/people/dhruv-kaunil/'>&nbsp;Facebook&nbsp;</a> as a Software Engineer in 2016 where I work on
                  computer vision and machine learning problems in Connectivity Lab
                  for the Smart Addresses Project.
                  <br/>

                  <br/>
                  My research interests include
                  inter-disciplinary applications of Machine Learning, Computer Vision,
                  and Natural Language Processing in Affective Computing and HCI
                  for the Blind Visually Impaired people.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }
}

export default App;
