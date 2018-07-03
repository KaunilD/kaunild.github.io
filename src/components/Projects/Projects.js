import React from 'react'

import './Projects.css'

class Projects extends React.Component {
  constructor (props) {
    super(props)
    this.state = {
    }
  }

  componentWillReceiveProps (nextProps) {
  }
  componentDidMount () {
  }
  render () {
    return (
      <div className='page-content'>
        <div className='page-content-inner'>
          <div className='page-title show-page-title'>
            <p data-value='PROJECTS'>
              PROJECTS
            </p>
          </div>
          <div className='page-descp show-page-descp'>
            <div className='page-items-container' ref={(element) => this.projects = element}>
              <div className='row'>
                <div className='col-30'>
                  <span className='h4 project-title'>COBRIX</span>
                  <span className='project-hint' >
                    Computer Programming for the blind and visually impaired.
                  </span>
                  <span className='project-short-descp'>
                    <ul>
                      <li>
                        Developed a machine learning pipeline which localized
                        and segmented lego blocks using a deep convolutional
                        neural network - Fast-RCNN.
                      </li>
                      <li>
                        Microsoft Imagine Cup Korea Semi Finalist.
                      </li>
                    </ul>
                  </span>
                  <p className='learn-more'>
                    <span>
                      Learn More
                    </span>
                  </p>
                </div>
                <div className='col-30'>
                  <span className='h4 project-title'>DISGUISED FACE IDENTIFICATION</span>
                  <span className='project-hint' >
                    Security System enhanced by Computer Vision and Machine Learning.
                  </span>
                  <span className='project-short-descp'>
                    <ul>
                      <li>
                        Developed an SVM classifier trained to seperate Masked and Unmasked faces.
                      </li>
                      <li>
                        Distances between the facial landmarks of the two image instances
                        were considered to perform the classification.
                      </li>
                    </ul>
                  </span>
                  <p className='learn-more'>
                    <span>
                      Learn More
                    </span>
                  </p>
                </div>
                <div className='col-30'>
                  <span className='h4 project-title'>MEDICAL REPORT DIGITIZER</span>
                  <span className='project-hint' >
                    Android Application based on OpenCV and Tesserract OCR.
                  </span>
                  <span className='project-short-descp'>
                    <ul>
                      <li>
                        This was an attempt at creating a data curation platform
                        by digitizing existing on-paper medical reports of the users.
                      </li>
                      <li>
                        The data then gathered is to be used in provind real time
                        alerts to the users regarding their health and the measures
                        that they should take inorder to improve them.
                      </li>
                    </ul>
                  </span>
                  <p className='learn-more'>
                    <span>
                      Learn More
                    </span>
                  </p>
                </div>
              </div>
              <div className='row'>
                <div className='col-30'>
                  <span className='h4 project-title'>APTITUDE TEST CMS</span>
                  <span className='project-hint' >
                    CMS specially designed for teachers and professors to
                    undertake quick weekly surprise tests for their students.
                  </span>
                  <span className='project-short-descp'>
                    <ul>
                      <li>
                        Entire setup contains a gigabyte motherboard,
                        tplink wifi router and Ubuntu linux configured as a web server.
                      </li>
                      <li>
                        The resulting systems forms a closed network connecting students
                        mobile devices and the server.
                      </li>
                      <li>
                        CMS developed in scratch using php, mysql and AngularJS.
                      </li>
                    </ul>
                  </span>
                  <p className='learn-more'>
                    <span>
                      Learn More
                    </span>
                  </p>
                </div>
                <div className='col-30'>
                  <span className='h4 project-title'>STANDALONE MULTIMEDIA STREAMER</span>
                  <span className='project-hint' >
                    A Raspberry Pi based multimedia streaming device.
                  </span>
                  <span className='project-short-descp'>
                    <ul>
                      <li>
                        RPi configured as a web server which creates a WiFi network
                        allowing 5 users to stream simultaneoulsy at speeds
                        2-5 mbps.
                      </li>
                      <li>
                        CMS developed in scratch using php, mysql and AngularJS.
                      </li>
                    </ul>
                  </span>
                  <p className='learn-more'>
                    <span>
                      Learn More
                    </span>
                  </p>
                </div>
                <div className='col-30'>
                  <span className='h4 project-title'>SMART TITLE</span>
                  <span className='project-hint' >
                    Implemented using the paper - https://arxiv.org/abs/1512.01712 in
                    chainer.
                  </span>
                  <p className='learn-more'>
                    <span>
                      Learn More
                    </span>
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

export default Projects
