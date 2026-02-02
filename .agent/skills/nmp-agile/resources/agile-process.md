# Problem

The NeMo Org ships a self-hosted platform of microservices provided as containers on a monthly release cadence. This presents some challenges that are different from many SaaS orgs that can have a more continuous release cycle. The team needs a good system for planning out these releases in a predictable and consistent way that allows for high throughput of product with minimal defects and minimal disruptions to release. This RFC outlines the NeMo approach to planning out releases and making sure they match these expectations. 

# Terminology & Definitions

| POR | Plan of Record. This is the specific list of vetted features that will ship for a given release date.  |
| :---- | :---- |
| **PRD**  | Product Requirements Doc. A clear set of customer requirements that provides a high degree of clarity and answers all questions that Engineering needs to start designing the architecture for a feature. May propose changes that will take multiple releases to complete. Should follow [this PRD template](https://nvidia.sharepoint.com/:w:/r/sites/EPG/_layouts/15/doc2.aspx?sourcedoc=%7BD111B612-15F6-4649-80A7-1FA0F8A8F172%7D&file=TEMPLATE%20Product%20Requirements%20Document.docx&action=default&mobileredirect=true).   |
| **RFC** | Request for Comments. An Engineering document where specific architectures are outlined that will solve a particular customer need. May or may not be based on a PRD, depending on the requirements. RFCs should follow [this process](https://docs.google.com/document/d/135n9Rdj6Uy2f4Yjr1cm7dguiPR0ccmqLjchVXsnoZtA/edit?tab=t.0).   |
|  |  |

# Background

NeMo releases have many dependent steps without much consistency on how they are planned. The team relies on a Plan of Record (POR) which serves as a vetted manifest of features that will go into a given release. This document should be finalized at least 6 weeks prior to the release, and 4 weeks prior to code freeze for that release. The POR is the result of several months of planning from Product Requirements, Architecture Design, and Development. However, there is no agreed upon method for producing the artifacts needed from each of these phases to make sure the POR has been properly planned. Product Requirements don’t have any standardization, and are often not filled in ahead of the POR meeting at all, and eng tickets are often not properly refined in time to give an accurate estimate. This results in a somewhat chaotic process around putting together the POR for a given release, and much confusion about what can or cannot be in a given release. 

# Goals

* A reasonably accurate estimation and velocity-tracking process that is low-impact for developers while providing metrics that allow managers to (1) evaluate that releases are on track and (2) validate that our staffing levels are appropriate.

* A prescriptive means of capturing high-level requirements effectively and then collaboratively decomposing them into stories/tasks that dev teams can implement against.

## Non-Goals

* Switching to a new ticket management system. For the purpose of this doc we will assume we are using GitLab and Aha to track issues.   
* The exact structure of Product Requirements. This should be aligned in a separate document. 

# Requirements

### REQ 1: Support Monthly Microservice Release Cadence.

The Agile process **MUST** enable consistent monthly microservice releases. All work should be associated with releases when applicable. The process **MUST** support regular weekly checkpoints within the release cycle to ensure a releasable increment is ready monthly. The process **MUST** allow for planning several releases ahead to support a consistent, sustainable rhythm of delivery. 

### REQ 2: Clear Requirements Pipeline

The proposed Agile process and chosen tooling **MUST** provide a clear mechanism for PM and other stakeholders to provide requirements with enough lead time for Engineering to write RFCs and produce a refined backlog for each release. The process **MUST** provide a mechanism for decomposing high-level requirements into SMART dev tasks. The process **MUST** provide clear guidelines for when and how product requirements are accepted to be scheduled for design and development. 

### REQ 3: Visibility (Scientists, Engineers, PMs).

The proposed Agile process and chosen tooling **MUST** provide shared, transparent visibility into project status, dependencies, and overall progress across all features included in each release. It **SHOULD** work effectively for all stakeholders: Engineering, Program Managers, and Product Managers, QA, etc. 

### REQ 4: Consistent and Low-Overhead Estimation.

The SDLC **MUST** use a consistent, low-overhead estimation technique. Everyone should use roughly the same estimation techniques outlined later in this document. The estimation process **SHOULD** be quick to implement during planning sessions, and be flexible enough to handle peoples time off without throwing estimates too far off. The estimation process **MUST** be accurate enough to reasonably estimate each monthly release. 

### REQ 5: Integrate Effectively with the Validation ProcessesQA Process.

The proposed SDLC **SHOULD** make it easy to track what is being validatedQA’d and allow touch points to fix any issues that come up in that process. This needs clear handoffs, robust bug tracking, comprehensive test case management, and transparent traceability. 

# Proposal

### Overview

The full SDLC of a release includes the following phases:

1. Requirements Gathering (PRDs)  
2. Architectural Designs (RFCs)  
3. Core Development   
4. QA / VDR

These phases will likely blend together. It’s appropriate to begin coding a Proof of Concept during the PRD phase to help understand the requirements, and it’s ok to begin writing the RFCs alongside the PRD in some cases. Development for non contentious issues can begin before the RFC has been approved to expedite the process. 

![][image1]

This process has monthly releases and development cycles, with each development cycle consisting of two sprints. Dev cycles for a release end with a code freeze roughly at the middle of the month. Code freeze marks the end of one development cycle and the beginning of another. As one cycle ends, we kick off the QA and VDR processes to prepare to release. The new development cycle starts with a finalized POR put together from a fully refined backlog. 

During each development cycle, we are simultaneously preparing the backlog for the next development cycle. This preparation is done through a mix of requirements gathering, document writing and reviewing, and backlog refinement. Tickets are continuously refined and estimated so that we are always prepared for each development cycle to begin with a fresh POR. 

- [ ] ## Phases of Delivery

#### Product Requirements

For PM led features, Product Requirements need to be completed in the form of a PRD with enough time for engineering to write RFCs and for those RFCs to be refined ahead of a finalized POR. Working backwards from these timelines, we should expect PRDs to be completed, reviewed, and accepted at least 2-3 months/releases in advance of when they are expected to release. Since this is done alongside existing delivery work for earlier releases, it should not slow down our overall delivery and should make for a more consistent and high quality production cadence over time. 

This phase isn’t complete until a high quality set of product requirements (a PRD for large features, or a detailed Aha ticket for small improvements) has been read by all relevant stakeholders and the PICs are aligned on the right customer experience. The requirements should be detailed enough to fully describe the desired user experience, no more and no less. That document should have a list of approvers who need to sign off on it to make sure that the requirements are clear enough for engineering to understand the problem and to properly design it. 

This phase is when PM involvement begins, not when it ends. For any product-led feature PM will be involved with every step of the process to make sure that the end product meets the customer’s expectations. They also will help coordinate how large features can be broken up across multiple releases, and how they will be sequenced as EA and GA releases. 

#### Architectural Design

RFCs should be completed roughly two months before we expect to ship a major feature. This phase consists of:

* Writing the RFC following the [NeMo RFC Process](https://docs.google.com/document/u/0/d/135n9Rdj6Uy2f4Yjr1cm7dguiPR0ccmqLjchVXsnoZtA/edit)  
* Reviewing any proposed API changes with the [AARB](https://docs.google.com/document/u/0/d/1jleQK7xRlO2WfXyb7829zODIJamZDQNTA-ovSi5J2wI/edit)  
* Making any changes required to satisfy feedback from AARB, PM, and all other stakeholders  
* Completing the review of the RFC and getting signoff  
* Breaking the work into refined Epics that include Story Point estimates. 

This phase ends with refined Epics that will be used to finalize the POR for the upcoming release.  

#### Core Development Cycle 

Each Release is associated with a Development Cycle of \~2 Sprints. This phase comes after Product and Architectural design, once we have a strong understanding of what we are building and have a fully refined backlog of issues . The majority of the development for the sprint takes place in this phase. It begins with the POR where we have agreed with stakeholders about what should go in the upcoming Release, and it ends when we start the QA and VDR process. 

Most work in a given Dev Cycle targets the upcoming release, but some features require more work than that and capacity can be set aside during any given cycle for a future release. If so, the upcoming release capacity needs to be adjusted for any developers working on features for future releases. 

#### QA

This phase can start on subcomponents as they are ready, and it’s best to get going early. However, at a minimum, this phase needs to be given 2 weeks before a release to allow time to fully test and to address any issues that come up. The QA process includes completing a Virtual Developer Review. 

# Implementation Details

## Sprints

We will align around 2 week sprints. This gives us 2 touch points per dev cycle, without adding the excessive meeting overhead of a 1 week sprint. Sprints should end with demos. 

## Release Tracking

As soon as we agree to build a feature we create an Epic for it. Tag that Epic with a placeholder release within the next 2-3 releases, along with appropriate phase – PRD for a PM led feature, or straight to RFC for an eng-led feature. We can use the [Releases Epic Board](https://gitlab-master.nvidia.com/groups/aire/-/epic_boards/479) to see a NeMo-wide view of the top level features that we expect to go into the upcoming releases. These Epic lists will correspond 1-1 with items in Aha that will be used for the POR. 

![][image2]

## Estimation and Capacity Planning

Features should get estimates as soon as possible. Epics or large issues can get low fidelity Tee shirt sizes right away, and should be refined into high fidelity Story Point estimations once designs/RFCs are ready and work is ready to begin.

#### Tee Shirt Sizes \- Epic Level 

##### Sizing

We will use simple tee-shirt sizing for low fidelity estimates. These estimates will not get any more fine-grained than full sprints. 

* **X-Small** \- 0.5 Developer Sprint   
* **Small** \- 1 Developer Sprint   
* **Medium** \- 2 Developer Sprints  
* **Large** \- 4 Developer Sprints  
* **X-Large** \- 6+ Developer Sprints

These low fidelity estimates are used to roughly place features into an appropriate release. They can be converted to and compared with Story Points by calculating the SP value of a Developer Sprint. 

##### Capacity Planning

For Tee Shirt Size capacity planning you first find your idealized capacity, and then discount it heavily to account for real world constraints. It’s up to the Manager to discount their capacity according to their team’s specific KTLO and on-call responsibilities, but an example baseline is provided below. 

1. **Idealized Capacity:** Your team size represents idealized dev sprints available at any given time. If your team has 10 members, then you have a theoretical capacity of 10 Dev Sprints at a time. However, this should be adjusted before estimating.   
2. **Adjusted Capacity:** Ideal capacity is never reflected in real world conditions, so the ideal needs to be heavily discounted to set realistic expectations. A decent starting point would be to discount by 50%, though this can be adjusted by the Engineering Manager based on their team's unique constraints (high KTLO, on-call responsibilities, etc.). The 50% discount comes from the following assumptions:   
   1. Subtract 10% to adjust for typical vacation time.   
   2. Subtract 20% for Reactive work, bug fixes and minor enhancements   
   3. Subtract 20% for non-coding work such as documentation, ad-hoc meetings, mentoring, etc. 

So a team of 10 may have a capacity of 5 Developer Sprints to put towards feature development. Since each Release has a development cycle of 2 sprints, they would have 10 Dev Sprints available for each Release, which would allow that team to estimate they’d complete 5 Medium features, or 2 Medium and 6 Small, etc. 

#### Story Points

For more precise estimates we will use standard Fibonacci Story Points, which represent a task’s **complexity**, rather than a time estimate. These estimates give us a higher confidence in our understanding of the work required to deliver on our upcoming release, which increases the likelihood that we will hit our POR goals.

We’ll have a maximum complexity score of 8, and anything more complex should be broken down into smaller tickets or given a Tee Shirt estimate. All tickets for a POR should be fully estimated before the POR is finalized, so that we can have a good estimate of what’s possible for that release. See [Story Points](#story-points) appendix for the definitions. In general we should have at least one Release worth of work (2 sprints) with Story Point estimates, so that we can have a high-confidence PoR for the next release at any given time. Beyond that (for Release+2 and onward), Story Point estimates have diminishing returns and the team should favor Tee Shirt Sizes. T

##### Sprint Planning

We’ll use a Trimmed Mean approach to estimate our story points. 

1. We will remove the highest and lowest values over the last 8 sprints, and average the remaining 6\. This is known as the **Trimmed Mean**, and will serve as our Sprint Capacity. This is more reliable than a true mean, because it removes outliers that may skew the number. Trimmed means are sometimes used as a “90% confidence interval”, though there is no statistical basis for that so I will avoid it.   
2. Calculate the value of a **Developer Sprint** by dividing your **Trimmed Mean** by the number of developers on the team. This can then be used to validate your Tee Shirt estimates and see if the estimates need to be moved. 

This type of estimation inherently has a cold-start problem, if you do not have 8 sprints worth of data. There isn’t much of a way around this. Before we get this data, we will just average all the data we have available until we have 8 sprints to reference. For instance when we have only one sprint, that will be our Sprint Capacity. When we have 2 or 3, it will be the average of these. Once we have 8 Sprints of trailing data for the team, we will convert to using the Trimmed Mean. 

##### Examples

* The team has 5 members, and has these values for the past 8 sprints: \[26, 31, 33, 33, 38, 44, 49, 88\]  
* **Sprint Capacity** \= mean(\[31, 33, 33, 38, 44, 49\]) \= **38.0**  
* **Developer Sprint** \= Sprint Capacity / Team Size \= 38 / 5 \= **8.55**

This means that you’d expect a **Large** feature to be equal to 8.55\*4 \~=**32 Story Points**. 

##### Notes of Caution

* All estimation techniques are used **only** to try to forecast the future, and not to retroactively judge performance.   
* It’s not possible to directly compare one team’s story point estimates to another’s, since each team estimates slightly differently. It’s ultimately the Engineering Manager who is responsible for speaking for and committing to these estimates.  

#### Ticket Hygiene

At a minimum, each ticket should have a Description and Acceptance Criteria in order to be put into a Refined state. This doesn’t have to be complicated, but must include a description and exit conditions. See [this simple issue template](#issue-template).  

## Meetings

##### Finalize PoR \- Release+1

At the start of each Development Cycle, we Finalized the POR for that release. We should review the epics going into this release and verify that we have capacity to meet those goals, making adjustments as needed. 

##### Kickoff Release+2 

Around the same time as the POR is finalized, we also have a kickoff meeting for the following release. This gives us plenty of time to discuss what’s realistic to get into that following release, and makes it obvious if we have unclear requirements that need to be clarified before we can have confidence of getting features ready for the release. 

##### Sprint Demos 

At the end of each sprint, we will do a NeMo wide Sprint Demo, so we can all see what we are working on. This also will focus the team to try to have as many demo-able tickets done at the end of each sprint. 

## Change Control 

The process lays out the happy path, but we operate in a highly volatile industry and can expect that sudden requirements come in that need us to change course. This should be baked into the process. Below are the mechanisms for handling scope changes.

* **New Product Requirements come in after the POR is finalized.** This represents a phase change back to the PRD phase, and by default should push the feature to a later release. If these changes need to be accommodated without slipping the release, then the teams should align with the PICs that the scope of the POR needs to change and decide with the team which other features will slip instead.  
* **A feature is urgently needed for the next release, but no PRD or RFCs exist**. A request like this should come directly from VP+, and should be taken seriously. We will work with the VP/E-Staff sponsor to identify which other features will slip to staff this new feature. This is not an alternate flow through the system, but an expedited one – we will still need high quality PRDs and RFCs, and need to refine and QA the work. Once the senior exec has signed off on the change of focus and we have staffed it, we should work very quickly on a PRD and RFC to allow maximum time for Development and QA. 

# Alternatives Considered

### Capacity Planning

I considered using confidence intervals calculated from the T-Distribution of the past N sprints, described [in this Agile presentation from the Triton team.](https://docs.google.com/presentation/d/1NOtaGHdBjcdAknpTWgMftY9NhM5gMdEw2j20dpMl2yQ/edit?slide=id.g13dec232289_0_609#slide=id.g13dec232289_0_609) However, I decided not to propose this method for two reasons:

1. This system looks at first to be statistically rigorous, and is attractive for the ability to give a level of confidence. It isn’t actually statistically correct however, as the confidence intervals would require that all of the data points be I.I.D (independent and identically distributed), but they are not. Tickets depend on each other, and sprint sizes influence each other. I’m skeptical of using methods that appear more precise than they are, so I am proposing a simple Trimmed Mean method.   
2. This system uses a definition of P0/P1/P2 label that would conflict with how other stakeholders think about it. P0 features typically mean features you would not ship without, not just the features that you have the highest confidence of shipping on time. To solve for this, we would have to add in even more labels for the level of confidence, and these would have to be updated every sprint to keep things aligned. 

# Appendix

### FAQ

#### This feels a little heavy \- is this Agile enough?

This process is agile for the nature of our business. Many of us associate Agile with the Scrum process that was popularized by web based startups in the 2000s and 2010s. The nature of these businesses allowed them to follow rapid 2 week release cycles (Sprints) that got product into customers hands for rapid review. That makes a ton of sense in that industry. 

What we are shipping are core platform infrastructure containers for a self-hosted AI Cloud. There is no way to iterate the same way as a small startup that is shipping a Rails app could in 2010\. That said, we can still come up with a measured release cycle that allows us to produce a steady stream of small releases at a good cadence. 

If you look at a single major feature, the cycle is relatively long and can take several months to release. If you step back and look at what the whole organization is producing, we should see that we have a very rapid flow of containers being shipped to customers. 

## Story Points  {#story-points}

| SP | Complexity | Ball Park Time | Example |
| :---- | :---- | :---- | :---- |
| 0 | Epic Level | \- | Epics should have an SP of 0\. The SP for the epic will be the sum of the issues in the epic. |
| 1 | Simple Task | \< 1 day | Simple refactoring with minimal code review, trivial bug, documentation authoring |
| 2 | Minor Task | 1-2 days | Small code changes, Code design, documentation, writing tests |
| 3 | Moderate Task | 2-4 days | Simple Multi-file code change w/ minimal dependencies, large test changes |
| 5 | Major Task | \< 1 week | Typical code change with dependencies, testing, and review |
| 8 | Difficult Task | 1-2 weeks | Large number of changes and dependencies, complex interactions, consider breaking into smaller stories |

## Issue Template {#issue-template}

**Description:** 

\<Brief 1-3 sentence description of the problem that needs to be solved, with relevant background context.\>

**Acceptance Criteria:**

* \<Condition 1 that must be met for the issue to be considered complete\>  
* \<Condition 2 that must be met for the issue to be considered complete\>

**Non-Goals**

* \<Conditions explicitly out of scope for this ticket\>

## Epic/Feature Template

**Customer Value:** 

\<Brief 1-3 sentence description of the problem that needs to be solved, with relevant background context.\>

**Dependent Features:**

* \<Title \- \[Epic Link to other feature that is dependent on this Epic’s Release\]\>

**Acceptance Criteria:**

* \<Condition 1 that must be met for the issue to be considered complete\>  
* \<Condition 2 that must be met for the issue to be considered complete\>

**Non-Goals**

* \<Conditions explicitly out of scope for this ticket\>
