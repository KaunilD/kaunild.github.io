import React from 'react'

import './Publications.css'

class Publications extends React.Component {
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
    return(
    <div className='page-content'>
      <div className='page-content-inner'>
        <div className='page-title show-page-title'>
          <p data-value='PUBLICATIONS'>
            PUBLICATIONS
          </p>
        </div>

        <div className='page-descp show-page-descp'>
          <div className='page-items-container' ref={(element) => this.publications = element}>
            <div className='row'>
              <div className='col-30'>
                <span className='h4'>
                  Robocodes: Towards Generative Street  Addresses from
                  Satellite Imagery.
                </span>
                <span className='project-hint' >
                  CVPR 2017 - July 21, 2017
                </span>
                <span className='project-short-descp'>
                  Ilke Demir, Forest Hughes, Aman Raj, Kleovoulos Tsourides,
                  Divyaa Ravichandran, Suryanarayana Murthy, Kaunil Dhruv,
                  Sanyam Garg, Jatin Malhotra, Barrett Doo, Grace Kermani,
                  Ramesh Raskar
                </span>
                <p className='learn-more'>
                  <span>
                    Learn More
                  </span>
                </p>
              </div>
              <div className='col-30'>
                <span className='h4'>Generative Street Addresses from Satellite Imagery.</span>
                <span className='project-hint' >
                  IJGI - March 8, 2018
                </span>
                <span className='project-short-descp'>
                  Ilke Demir, Forest Hughes, Aman Raj, Kaunil Dhruv,
                  Suryanarayana Murthy Muddala, Sanyam Garg, Barrett Doo,
                  Ramesh Raskar
                </span>
                <p className='learn-more'>
                  <span>
                    Learn More
                  </span>
                </p>
              </div>
              <div className='col-30'>
                <span className='h4'>
                  A Holistic Framework for Addressing the World using
                  Machine Learning
                </span>
                <span className='project-hint' >
                  CVPR 2018 - June 18, 2018
                </span>
                <span className='project-short-descp'>
                  Ilke Demir, Forest Hughes, Aman Raj, Kaunil Dhruv,
                  Suryanarayana Murthy Muddala, Sanyam Garg, Barrett Doo
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

export default Publications
