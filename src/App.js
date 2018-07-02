import React, { Component } from 'react'
import ReactDOM from 'react-dom'
import ScriptString from './components/ScriptString'
import About from './components/About'
import Publications from './components/Publications'
import Projects from './components/Projects'
import WebGLSplash from './components/WebGLSplash'
import './App.css'

var smoothScroll = require('smoothscroll')

class App extends Component {

  constructor (props) {
    super(props)
    this.state = {
      title: 'Kaunil Dhruv',
      subTitle: 'SOFTWARE ENGINEER',
      socialLinks: false,
      page1Descp: false,
      page1Seperator: false,
      page1Overlay: false,
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
    if (scrollTop >= ReactDOM.findDOMNode(this.page1).clientHeight) {
      this.setState({page2Overlay: true})
    }
    if (scrollTop >= ReactDOM.findDOMNode(this.page1).clientHeight + ReactDOM.findDOMNode(this.page2).clientHeight) {
      this.setState({page3Overlay: true})
    }

    if (scrollTop >= ReactDOM.findDOMNode(this.page1).clientHeight + ReactDOM.findDOMNode(this.page2).clientHeight + ReactDOM.findDOMNode(this.page3).clientHeight) {
      this.setState({page4Overlay: true})
    }
    if (scrollTop >= ReactDOM.findDOMNode(this.page1).clientHeight + ReactDOM.findDOMNode(this.page2).clientHeight + ReactDOM.findDOMNode(this.page3).clientHeight + ReactDOM.findDOMNode(this.page5).clientHeight) {
      this.setState({page5Overlay: true})
    }
  }

  handleLinkClick (event, target) {
    event.preventDefault()
    smoothScroll(
      target,
      500,
      null,
      this.content
    )
  }

  render () {
    return (
      <div>
      <WebGLSplash />
      <div className='content-wrapper'>
        <div className='sidebar'>
          <div className='profile'>
            <div
              className={this.state.profileText ? 'profile-text slideup-profile-text' : 'profile-text'}
              ref={(element) => this.profileText = element}
              >
              <div className='title'>
                <span className='title-hi'>Hi I'm</span>
                <ScriptString onComplete={() => this.toggleSocialLinks()} string="KAUNIL DHRUV"/>
              </div>
            </div>

            <div className={this.state.menu ? 'menu show-menu' : 'menu'}>
              <ul className='menu-main'>
                <li>
                  <a onClick={(e) => this.handleLinkClick(e, this.page1)}>
                    About
                  </a>
                </li>
                <li>
                  <a onClick={(e) => this.handleLinkClick(e, this.page2)}>Publications</a>
                </li>
                <li>
                  <a onClick={(e) => this.handleLinkClick(e, this.page3)}>Projects</a>
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
                  <a onClick={(e) => this.handleLinkClick(e, this.page4)}>Experiments</a>
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
                  <a href='assets/resume.pdf' target='blank'>Resume</a>
                </li>
                <li>
                  <a onClick={(e) => this.handleLinkClick(e, this.page5)}>Contact</a>
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
              <p>Built with <img src='react.png' /></p>
            </div>
          </div>


        </div>
        <div className='content' onScroll={this.onScroll} ref={(element) => this.content = element}>
          <div className='page-1' ref={(element) => this.page1 = element}>
            <About  ref={(element) => this.page1 = element}
              page1Descp={this.state.page1Descp}
              page1Seperator={this.state.page1Seperator}
              page1Overlay={this.state.page1Overlay}
            />
          </div>
          <div className='page-2' ref={(element) => this.page2 = element}>
            <Publications
            />
          </div>

          <div className='page-3' ref={(element) => this.page3 = element}>
            <Projects />
          </div>

          <div className='page-4' ref={(element) => this.page4 = element}>
            <div className='page-content'>
              <div className='page-content-inner'>
                <div className='page-title show-page-title'>
                  <p data-value='EXPERIMENTS'>
                    EXPERIMENTS
                  </p>
                </div>
                <div className='page-descp show-page-descp'>

                  <div className='page-items-container' ref={(element) => this.experiments = element}>

                    <div className='page-item-full' id='01'>
                      <div>
                        <span className='h4'>MACHINE LEARNING</span>
                      </div>
                    </div>
                    <div className='page-item-full' id='02'>
                      <div>
                        <span className='h4'>REACT JS</span>
                      </div>
                    </div>
                    <div className='page-item-full' id='03'>
                      <div>
                        <span className='h4'>COMPUTER VISION</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className='page-5' ref={(element) => this.page5 = element}>
            <div className='page-content'>
              <div className='page-content-inner'>
                <div className='page-title show-page-title'>
                  <p data-value='CONTACT'>
                    CONTACT
                  </p>
                </div>
                <div className='page-descp show-page-descp'>
                  <div className='page-items-container' ref={(element) => this.contact = element}>

                  </div>
                </div>
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
