import React, { Component } from 'react'
import ReactDOM from 'react-dom'
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
      page1Title: false,
      page1Overlay: false,
      page2Overlay: false,
      page3Overlay: false,
      page4Overlay: false,
      profileText: false,
      menu: false
    }

    this.onScroll = this.onScroll.bind(this)
  }

  componentDidMount () {
  }

  toggleSocialLinks () {
    this.setState({socialLinks: true, profileText: true})
    setTimeout(this.initPage1Visibility(), 1000)
  }

  initPage1Visibility () {
    let parent = this
    setTimeout(function () {
      parent.setState({menu: true, page1Overlay: true, page1Title: true, page1Descp: true, page1Seperator: true})
    }, 300)
  }

  onScroll () {
    let scrollTop = ReactDOM.findDOMNode(this.content).scrollTop
    console.log(scrollTop, ReactDOM.findDOMNode(this.page1).clientHeight)
    if (scrollTop >= ReactDOM.findDOMNode(this.page1).clientHeight) {
      this.setState({page2Overlay: true})
    }
    if (scrollTop >= ReactDOM.findDOMNode(this.page1).clientHeight + ReactDOM.findDOMNode(this.page2).clientHeight) {
      this.setState({page3Overlay: true})
    }

    if (scrollTop >= ReactDOM.findDOMNode(this.page1).clientHeight + ReactDOM.findDOMNode(this.page2).clientHeight + ReactDOM.findDOMNode(this.page3).clientHeight) {
      this.setState({page4Overlay: true})
    }
  }

  render () {
    return (
      <div className='content-wrapper'>
        <div className='sidebar'>
          <div className='profile'>
            <div
              className={this.state.profileText ? 'profile-text slideup-profile-text' : 'profile-text'}
              ref={(element) => this.profileText = element}
              >
              <div className='title'>
                KAUNIL DHRUV
              </div>
              <div className='subtitle'><span><ScriptString onComplete={() => this.toggleSocialLinks()} string={this.state.subTitle}/></span></div>
            </div>

            <div className={this.state.menu ? 'menu show-menu' : 'menu'}>
              <ul className='menu-main'>
                <li>
                  <a href='#'>
                    About
                  </a>
                </li>
                <li>
                  <a>Publications</a>
                </li>
                <li>
                  <a href='#'>Projects</a>
                  <ul className='sub-menu'>
                    <li>
                      COBRIX
                    </li>
                    <li>
                      DISGUISED FACE DETECTION
                    </li>
                    <li>
                      SMART TITLE
                    </li>
                    <li>
                      L.M.S
                    </li>
                    <li>
                      DIGITIZING MEDICAL REPORTS
                    </li>
                    <li>
                      STANDALONE MULTIMEDIA BOX
                    </li>
                    <li>
                      APTITIDE CMS
                    </li>
                  </ul>
                </li>
                <li>
                  <a href='#'>Experiments</a>
                  <ul className='sub-menu'>
                    <li>
                      REACT JS
                    </li>
                    <li>
                      MACHINE LEARNING
                    </li>
                    <li>
                      COMPUTER VISION
                    </li>
                  </ul>
                </li>

                <li>
                  <a>Resume</a>
                </li>
                <li>
                  <a>Contact</a>
                </li>
              </ul>
            </div>



            <div ref={(elt) => this.socialLinks = elt} className={this.state.socialLinks ? 'social-links show-social-links' : 'social-links'}>
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
            <div className='notices'>
              <p>Built with <img src='favicon.ico' /></p>
            </div>
          </div>


        </div>
        <div className='content' onScroll={this.onScroll} ref={(element) => this.content = element}>
          <div className='page-1' ref={(element) => this.page1 = element}>
            <div className={!this.state.page1Overlay ? 'page-1-overlay' : 'page-1-overlay hide-page-1-overlay'} />
            <div className='page-content'>
              <div className='page-content-inner'>
                <div className={this.state.page1Title ? 'page-title show-page-title' : 'page-title'}>
                  <p>
                    !Hi
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

          <div className='page-2' ref={(element) => this.page2 = element}>
            <div className={!this.state.page2Overlay ? 'page-2-overlay' : 'page-2-overlay hide-page-2-overlay'} />
            <div className='page-content'>
              <div className='page-content-inner'>
                <div className={this.state.page1Title ? 'page-title show-page-title' : 'page-title'}>
                  <p>
                    Publications
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

          <div className='page-3' ref={(element) => this.page3 = element}>
            <div className={!this.state.page3Overlay ? 'page-3-overlay' : 'page-3-overlay hide-page-3-overlay'} />
          </div>

          <div className='page-4' ref={(element) => this.page4 = element}>
            <div className={!this.state.page4Overlay ? 'page-4-overlay' : 'page-4-overlay hide-page-4-overlay'} />
          </div>
        </div>
      </div>
    )
  }
}

export default App;
