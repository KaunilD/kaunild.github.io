import React, { Component } from 'react'
import Particles from 'react-particles-js'

import ScriptString from './components/ScriptString'

import './App.css'


class App extends Component {

  constructor (props) {
    super(props)
    this.state = {
      title: 'Kaunil Dhruv',
      subTitle: 'SOFTWARE ENGINEER',
      socialLinks: false,
      page1Descp: false,
      page1Seperator: false,
      page1Title: false
    }
  }

  componentDidMount () {
  }

  toggleSocialLinks () {
    let parent = this
    this.setState({socialLinks: true})
    setTimeout(this.initPage1Visibility(), 100)
  }

  initPage1Visibility () {
    let parent = this
    setTimeout( function () {
      parent.setState({page1Title: true})
      setTimeout( function () {
        parent.setState({page1Seperator: true})
        setTimeout( function () {
          parent.setState({page1Descp: true})
        }, 1500)
      }, 900)
    }, 300)
  }

  render () {
    return (
      <div className='content-wrapper'>
        <div class='sidebar'>
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
            className='particles-wrapper'
            canvasClassName='particles'
          />
          <div className='profile'>
            <div className='profile-text'>
              <div className='title'>
                KAUNIL DHRUV
              </div>
              <div className='subtitle'><span><ScriptString onComplete={() => this.toggleSocialLinks()} string={this.state.subTitle}/></span></div>
            </div>

            <div className={this.state.socialLinks ? 'social-links show-social-links' : 'social-links'}>
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


        </div>
        <div class='content'>

          <div className='page-1'>
            <div className='page-content'>
              <div className='page-content-inner'>
                <div className={this.state.page1Title ? 'page-title show-page-title' : 'page-title'}>
                  <p>
                    About Me
                  </p>
                </div>
                <div className={this.state.page1Seperator ? 'page-seperator show-page-seperator' : 'page-seperator'} />
                <div className={this.state.page1Descp ? 'page-descp show-page-descp' : 'page-descp'}>
                  <p>
                  I joined <a target='blank' className='facebook' href='https://research.fb.com/people/dhruv-kaunil/'>&nbsp;Facebook&nbsp;</a> as a Software Engineer in 2016 where I work on
                  computer vision and machine learning problems in Connectivity Lab
                  for the Smart Addresses Project.
                  <br/><br/>
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
